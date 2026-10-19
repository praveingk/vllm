# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NixlPushConnector into a mixed-memory (HiSparse) consumer.

A HiSparse decode keeps the sparse-MLA cache in a host-resident (DRAM) group
and the indexer cache in a device group. The producer keeps both on the GPU,
so it WRITEs region by region and splits each WRITE across the consumer's
DRAM and device dlists, notifying only once both have landed.
"""

import queue
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    TPMapping,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)

pytestmark = pytest.mark.cpu_test

D = "D"
NUM_BLOCKS = 100
BLOCK_SIZE = 64
# Region descriptor bases: one region per layer, NUM_BLOCKS descs each.
HOST, INDEXER = 0, 1
DRAM_HANDLE, DEVICE_HANDLE, LOCAL_HANDLE = 500, 600, 700


def _push_worker(remote_ranks=(0,), dram_ranks=(0,)) -> NixlPushConnectorWorker:
    """P worker with a handshaken mixed-memory D: real descriptor maths,
    only the NIXL transport mocked."""
    w = object.__new__(NixlPushConnectorWorker)
    w.shutdown = lambda: None  # skeleton worker: silence __del__
    w.engine_id = "P"
    w.world_size = w.transfer_tp_size = 1
    w.block_size = BLOCK_SIZE
    w.use_mla, w._has_mamba, w._is_csa_linear = True, False, False
    w.dcp_size, w.dcp_rank = 1, 0
    w._physical_blocks_per_logical_kv_block = 1
    w._transfer_layer_group_ids = ()
    w._group_spec_types = (MLAAttentionSpec, MLAAttentionSpec)
    # P: GPU-only, two device groups (sparse MLA, indexer), one region each.
    w.region_group_ids = [0, 1]
    w._uses_region_group_mapping = True
    w.num_regions = 2
    # D (HiSparse): the same two layers, but only the host group is
    # transferable alongside the indexer, and a resident group comes first.
    w.dst_region_group_ids = {D: [0, 1]}
    w.dst_uses_region_group_mapping = {D: True}
    w.dst_num_blocks = {D: NUM_BLOCKS, "P": NUM_BLOCKS}
    w.dst_region_num_blocks = {D: [NUM_BLOCKS] * 2, "P": [NUM_BLOCKS] * 2}
    w.dst_region_mem_types = {D: ["DRAM", "VRAM"]}

    w.transfer_topo = MagicMock()
    w.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_tp_size=len(remote_ranks),
        remote_block_size=BLOCK_SIZE,
        remote_physical_blocks_per_logical=1,
    )
    w.transfer_topo.tp_ratio.return_value = -len(remote_ranks)
    w.transfer_topo.block_size_ratio.return_value = 1
    w.tp_mappings = {D: TPMapping(((0,), (0,)), (0,), {0: 0}, 0)}

    w.nixl_wrapper = MagicMock()
    w.nixl_wrapper.make_prepped_xfer.side_effect = iter(range(1000, 2000))
    w.xfer_stats = MagicMock()
    w._log_failure = MagicMock()
    w._remote_agents = {D: {(0, r): f"D-rank{r}" for r in remote_ranks}}
    w.src_xfer_handles_by_block_size = {BLOCK_SIZE: LOCAL_HANDLE}
    w.dst_xfer_side_handles = defaultdict(dict)
    w._dst_dram_xfer_side_handles = defaultdict(dict)
    w._dst_desc_is_dram = {}
    w._dst_desc_pos = {}
    w._sending_transfers = defaultdict(list)
    w._send_failures = set()
    w._sending_transfers_lock = threading.Lock()
    w._pending_push_notifs = {}
    w._pending_push_notifs_staged = []

    # Register D's per-rank dlists through the real split.
    blocks_data = np.zeros((2 * NUM_BLOCKS, 3), dtype=np.uint64)
    blocks_data[:, 2] = 3  # remote GPU id
    w.nixl_wrapper.prep_xfer_dlist.side_effect = lambda agent, descs: (
        DRAM_HANDLE if descs == "DRAM" else DEVICE_HANDLE
    )
    w.nixl_wrapper.get_xfer_descs.side_effect = lambda data, mem_type: mem_type
    w._TRANSFER_MODE = "push"
    for rank in remote_ranks:
        w._register_mixed_remote_xfer_handles(
            D, rank, f"D-rank{rank}", blocks_data, rank not in dram_ranks
        )
    return w


def _meta(local, remote, *, num_tokens=None, num_computed=()) -> ReqMeta:
    return ReqMeta(
        local_block_ids=local,
        local_physical_block_ids=local,
        tp_size=1,
        remote=RemoteMeta(
            block_ids=remote,
            host="",
            port=0,
            engine_id=D,
            request_id="d-req",
            num_tokens=num_tokens,
            num_computed_blocks=num_computed,
        ),
    )


def _writes(w):
    """(remote dlist, local desc ids, remote dlist positions) per WRITE."""
    return [
        (c.args[3], list(c.args[2]), list(c.args[4]))
        for c in w.nixl_wrapper.make_prepped_xfer.call_args_list
    ]


def test_mixed_remote_registration_splits_dram_and_device_dlists():
    """The DRAM half of D's descs gets its own dlist on CPU device 0."""
    w = _push_worker()
    dram_call, device_call = w.nixl_wrapper.get_xfer_descs.call_args_list
    dram_data, dram_type = dram_call.args
    device_data, device_type = device_call.args
    assert (dram_type, device_type) == ("DRAM", "VRAM")
    assert len(dram_data) == len(device_data) == NUM_BLOCKS
    assert set(dram_data[:, 2]) == {0} and set(device_data[:, 2]) == {3}
    assert w._dst_dram_xfer_side_handles[D] == {0: DRAM_HANDLE}
    assert w.dst_xfer_side_handles[D] == {0: DEVICE_HANDLE}
    assert w._dst_desc_is_dram[D].tolist() == [True] * 100 + [False] * 100
    # Each descriptor is indexed within its own memory type's dlist.
    assert w._dst_desc_pos[D].tolist() == list(range(100)) * 2


def test_push_splits_write_and_defers_notif_until_all_done():
    """One WRITE per memory type, no notif on either: it goes out only once
    get_transfer_results sees every handle of the request finish."""
    w = _push_worker()
    meta = _meta([[10, 11, 12], [20, 21, 22]], [[30, 31, 32], [40, 41, 42]])

    w._xfer_blocks_for_req("p-req", meta)

    assert _writes(w) == [
        (DRAM_HANDLE, [10, 11, 12], [30, 31, 32]),
        (DEVICE_HANDLE, [120, 121, 122], [40, 41, 42]),
    ]
    assert all(
        "notif_msg" not in c.kwargs
        for c in w.nixl_wrapper.make_prepped_xfer.call_args_list
    )
    assert w._sending_transfers["p-req"] == [1000, 1001]
    assert w._pending_push_notifs == {"p-req": [("D-rank0", b"d-req:1")]}
    w.nixl_wrapper.send_notif.assert_not_called()


def test_push_writes_only_uncached_tail_by_token_position():
    """D's prefix hit and trailing padding are paired by token position,
    region by region (host and indexer may differ in cached pages)."""
    w = _push_worker()
    # 200 tokens = 4 pages; D has 1 host page and 0 indexer pages cached and
    # allocated one padding page in each region.
    local = [[10, 11, 12, 13], [20, 21, 22, 23]]
    remote = [[31, 32, 33, 39], [40, 41, 42, 43, 49]]
    meta = _meta(local, remote, num_tokens=200, num_computed=(1, 0))

    w._xfer_blocks_for_req("p-req", meta)

    assert _writes(w) == [
        (DRAM_HANDLE, [11, 12, 13], [31, 32, 33]),
        (DEVICE_HANDLE, [120, 121, 122, 123], [40, 41, 42, 43]),
    ]


def test_push_replicates_mla_to_every_decode_rank():
    """MLA is replicated over D's TP ranks: each handshaken rank gets both
    WRITEs and its own deferred notif."""
    w = _push_worker(remote_ranks=(0, 1), dram_ranks=(0, 1))
    w._xfer_blocks_for_req("p-req", _meta([[10], [20]], [[30], [40]]))

    assert len(_writes(w)) == 4
    assert w._pending_push_notifs["p-req"] == [
        ("D-rank0", b"d-req:1"),
        ("D-rank1", b"d-req:1"),
    ]


def test_push_into_shared_host_pool_writes_dram_once():
    """Ranks sharing rank 0's HiSparse pool registered no DRAM: only rank 0
    receives the host part, the others get just their device part."""
    w = _push_worker(remote_ranks=(0, 1, 2), dram_ranks=(0,))
    w._xfer_blocks_for_req("p-req", _meta([[10], [20]], [[30], [40]]))

    assert [handle for handle, *_ in _writes(w)] == [
        DRAM_HANDLE,
        DEVICE_HANDLE,
        DEVICE_HANDLE,
        DEVICE_HANDLE,
    ]
    assert [agent for agent, _ in w._pending_push_notifs["p-req"]] == [
        "D-rank0",
        "D-rank1",
        "D-rank2",
    ]


def test_dram_only_push_to_non_owner_rank_notifies_immediately():
    """A non-owner rank with nothing on the device still must hear the push
    complete; it has no WRITE of its own to wait on."""
    w = _push_worker(remote_ranks=(0, 1), dram_ranks=(0,))
    spec = ReadSpec(1, [[10], []], [[30], []], True)
    assert (
        w._xfer_blocks_mixed(
            read_spec=spec,
            dst_engine_id=D,
            request_id="p-req",
            remote_request_id="d-req",
            local_xfer_side_handle=LOCAL_HANDLE,
        )
        == []
    )
    w.nixl_wrapper.send_notif.assert_called_once_with("D-rank1", notif_msg=b"d-req:1")


def _completed(w, states):
    """Poll the P worker's push accounting with the given WRITE states."""
    w.nixl_wrapper.check_xfer_state.side_effect = states
    w.nixl_wrapper.get_xfer_telemetry.return_value = None
    w._reqs_to_send = {"p-req": float("inf")}
    w._reqs_to_process = {"p-req"}
    w.consumer_notification_counts_by_req = defaultdict(int)
    w._push_writer_wake = threading.Event()
    w._evict_finished_inbox = queue.Queue()
    # Isolate the push accounting from the base-class poll.
    base_results = SimpleNamespace(finished_sending=set(), finished_recving=set())
    with patch.object(
        NixlBaseConnectorWorker, "get_transfer_results", return_value=base_results
    ):
        return w.get_transfer_results()


def test_deferred_notif_sent_once_all_writes_done():
    w = _push_worker()
    w._xfer_blocks_for_req("p-req", _meta([[10], [20]], [[30], [40]]))

    results = _completed(w, ["DONE", "PROC"])
    assert results.finished_sending == set()
    w.nixl_wrapper.send_notif.assert_not_called()

    results = _completed(w, ["DONE"])
    assert results.finished_sending == {"p-req"}
    w.nixl_wrapper.send_notif.assert_called_once_with("D-rank0", notif_msg=b"d-req:1")
    assert w._pending_push_notifs == {}


def test_failed_split_write_never_notifies():
    """A failed half must not tell D the KV is complete: D fails the request
    on its registration watchdog instead of decoding partial KV."""
    w = _push_worker()
    w._TRANSFER_MODE = "push"
    w._recving_metadata = {}
    w._xfer_blocks_for_req("p-req", _meta([[10], [20]], [[30], [40]]))

    results = _completed(w, ["ERR", "DONE"])
    assert results.finished_sending == set()
    w.nixl_wrapper.send_notif.assert_not_called()
    assert w._pending_push_notifs == {}


def test_submission_failure_keeps_unreleasable_handles():
    """If the second WRITE fails to post, the first may already be in
    flight; a handle that will not release stays tracked."""
    w = _push_worker()
    w.nixl_wrapper.transfer.side_effect = [None, RuntimeError("post")]
    w._handle_failed_transfer = MagicMock(side_effect=[False, True])

    w._xfer_blocks_for_req("p-req", _meta([[10], [20]], [[30], [40]]))

    assert w._sending_transfers["p-req"] == [1000]
    assert "p-req" in w._send_failures
    assert w._pending_push_notifs == {}


def test_pull_still_rejects_mixed_memory_producer():
    w = _push_worker(remote_ranks=())
    w._TRANSFER_MODE = "pull"
    with pytest.raises(NotImplementedError, match="mixed-memory producer"):
        w._register_mixed_remote_xfer_handles(
            D, 0, "D-rank0", np.zeros((200, 3), dtype=np.uint64), False
        )


def test_cleanup_releases_dram_dlists():
    w = _push_worker(remote_ranks=(0, 1), dram_ranks=(0,))
    w.nixl_wrapper.release_dlist_handle.reset_mock()
    w._release_mixed_remote_xfer_handles(D)
    w.nixl_wrapper.release_dlist_handle.assert_called_once_with(DRAM_HANDLE)
    assert D not in w._dst_desc_pos and D not in w._dst_desc_is_dram


@pytest.mark.parametrize("cached", [0, 1])
def test_decode_zeroes_only_unwritten_padding(cached):
    """D zeroes the pages past the pushed token range, per region."""
    w = object.__new__(NixlPushConnectorWorker)
    w.shutdown = lambda: None
    w.block_size = BLOCK_SIZE
    w._physical_blocks_per_logical_kv_block = 1
    w.region_group_ids = [0, 1]
    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    w.kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["resident"], spec, enable_kv_transfer=False),
            KVCacheGroupSpec(["host"], spec, host_resident=True),
            KVCacheGroupSpec(["indexer"], spec),
        ],
    )
    # 200 tokens = 4 pages; the host group has ``cached`` pages from the
    # prefix cache, and each region one padding page.
    host = list(range(30 + cached, 35))
    meta = ReqMeta(
        local_block_ids=[host, list(range(40, 45))],
        local_physical_block_ids=[host, list(range(40, 45))],
        tp_size=1,
        local_num_computed_blocks=(0, cached, 0),
        remote=RemoteMeta([], "", 0, "P", "p-req", num_tokens=200),
    )
    assert w._unwritten_region_blocks(meta) == [[34], [44]]


# ----------------------------------------------------------------- #
#  DCP-sharded producer (prefill PCP/DCP > 1) into an unsharded D     #
# ----------------------------------------------------------------- #


def _dcp_shard(dcp_rank, dcp_size=4, remote_ranks=(0, 1)):
    """P shard ``dcp_rank`` of a DCP-sharded prefill (TP1, PCP=DCP=dcp_size),
    handshaken with every rank of an unsharded HiSparse decode."""
    w = _push_worker(remote_ranks=remote_ranks, dram_ranks=remote_ranks)
    w.dcp_size, w.dcp_rank = dcp_size, dcp_rank
    w.transfer_tp_size = dcp_size
    # The DCP-sharded P only handshakes its own source rank for MLA.
    w.tp_mappings = {D: TPMapping(((0,), (0,)), (0,), {0: 0}, 0)}
    w.transfer_topo.tp_ratio.return_value = 1
    return w


def _shard_pages(dcp_rank, dcp_size, num_pages, base):
    """P shard's local page ids, one per global position it owns."""
    return [base + g for g in range(num_pages) if g % dcp_size == dcp_rank]


@pytest.mark.parametrize("dcp_rank", range(4))
@pytest.mark.parametrize("cached", [0, 3, 6])
def test_dcp_shard_writes_only_its_token_slice(dcp_rank, cached):
    """Shard k WRITEs D's pages at global positions g % dcp == k, into every
    D rank, skipping D's prefix-cache hit (which may differ per region)."""
    num_pages = 10  # 640 tokens at block size 64
    w = _dcp_shard(dcp_rank)
    # D registered its uncached pages (host region skips ``cached``, the
    # indexer region has no hit) plus one padding page per region.
    remote = [
        [10 + g for g in range(cached, num_pages + 1)],
        [30 + g for g in range(0, num_pages + 1)],
    ]
    local = [
        _shard_pages(dcp_rank, 4, num_pages, 50),
        _shard_pages(dcp_rank, 4, num_pages, 70),
    ]
    meta = _meta(
        local, remote, num_tokens=num_pages * BLOCK_SIZE, num_computed=(cached, 0)
    )

    w._xfer_blocks_for_req("p-req", meta)

    host = [(50 + g, 10 + g) for g in range(cached, num_pages) if g % 4 == dcp_rank]
    indexer = [(70 + g, 30 + g) for g in range(num_pages) if g % 4 == dcp_rank]
    writes = _writes(w)
    # One DRAM (host) and one device (indexer) WRITE to each of the 2 D ranks.
    assert [handle for handle, *_ in writes] == [DRAM_HANDLE, DEVICE_HANDLE] * 2
    for handle, local_ids, remote_pos in writes:
        if handle == DRAM_HANDLE:
            assert list(zip(local_ids, remote_pos)) == host
        else:
            # Local indexer region descs follow the host region (offset 100).
            assert list(zip(local_ids, remote_pos)) == [
                (lid + NUM_BLOCKS, rid) for lid, rid in indexer
            ]
    # The notif tells D how many shards will report: the transfer TP size.
    assert w._pending_push_notifs["p-req"] == [
        ("D-rank0", b"d-req:4"),
        ("D-rank1", b"d-req:4"),
    ]


def test_dcp_shard_with_no_pages_still_notifies_every_decode_rank():
    """A prompt shorter than the shard's first page leaves it nothing to
    WRITE; D still waits for its notif, and P must still free the request."""
    w = _dcp_shard(dcp_rank=3)
    meta = _meta(
        [[], []],
        [[100, 101], [200, 201]],
        num_tokens=2 * BLOCK_SIZE,
        num_computed=(0, 0),
    )

    w._xfer_blocks_for_req("p-req", meta)

    w.nixl_wrapper.make_prepped_xfer.assert_not_called()
    assert [c.args for c in w.nixl_wrapper.send_notif.call_args_list] == [
        ("D-rank0",),
        ("D-rank1",),
    ]
    assert {
        c.kwargs["notif_msg"] for c in w.nixl_wrapper.send_notif.call_args_list
    } == {b"d-req:4"}
    assert w._sending_transfers["p-req"] == []


def test_dcp_shard_all_cached_on_decode_still_notifies():
    """D's prefix hit covering every page of this shard is the same case."""
    w = _dcp_shard(dcp_rank=0)
    meta = _meta(
        [[300], [400]], [[], [200]], num_tokens=2 * BLOCK_SIZE, num_computed=(2, 1)
    )
    w._xfer_blocks_for_req("p-req", meta)
    w.nixl_wrapper.make_prepped_xfer.assert_not_called()
    assert w.nixl_wrapper.send_notif.call_count == 2


def test_dcp_push_without_decode_counts_is_rejected():
    """Pairing a shard by list position would scramble tokens silently."""
    w = _dcp_shard(dcp_rank=1)
    with pytest.raises(NotImplementedError, match="DCP"):
        w._xfer_blocks_for_req("p-req", _meta([[300], [400]], [[100], [200]]))


def test_dcp_shard_handshakes_every_decode_rank():
    w = _dcp_shard(dcp_rank=2)
    w.kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    assert w._handshake_target_ranks(remote_tp_size=8, remote_dcp_size=1) == list(
        range(8)
    )


def test_unsharded_producer_keeps_read_side_handshake_targets():
    w = _push_worker()
    w.kv_transfer_config = SimpleNamespace(is_kv_producer=True)
    w.transfer_topo.handshake_target_ranks.return_value = [0]
    assert w._handshake_target_ranks(remote_tp_size=8, remote_dcp_size=1) == [0]


def _decode_worker(tp_size=8):
    """D rank of an unsharded TP decode, tracking one pushed request."""
    w = object.__new__(NixlPushConnectorWorker)
    w.shutdown = lambda: None
    w.world_size = w.transfer_tp_size = tp_size
    w.transfer_topo = MagicMock()
    w._pending_completion_notifs = queue.Queue()
    w._reqs_to_send, w._reqs_to_process = {}, set()
    w.consumer_notification_counts_by_req = defaultdict(int)
    w._recving_transfers = defaultdict(list)
    w.xfer_stats = MagicMock()
    return w


@pytest.mark.parametrize("dcp_size", [1, 8])
def test_decode_waits_for_one_notif_per_dcp_shard(dcp_size):
    """Each DCP shard writes a distinct slice: D is complete only after all
    of them report, never on the first."""
    w = _decode_worker()
    w._recving_metadata = {"d-req": SimpleNamespace(pp_size=1, dcp_size=dcp_size)}
    for _ in range(dcp_size - 1):
        w._pending_completion_notifs.put(f"d-req:{dcp_size}".encode())
    w._get_new_notifs()
    assert "d-req" not in w._recving_transfers

    w._pending_completion_notifs.put(f"d-req:{dcp_size}".encode())
    w._get_new_notifs()
    assert "d-req" in w._recving_transfers


def test_decode_registration_carries_producer_dcp_size():
    """D must handshake every P shard, so it needs P's DCP size."""
    from .test_nixl_push_connector import _BlocksMock, _make_request, _stub_sw_clipping
    from .utils import make_nixl_push_scheduler

    sched = make_nixl_push_scheduler()
    _stub_sw_clipping(sched)
    request = _make_request(request_id="req-d")
    request.kv_transfer_params["dcp_size"] = 8
    sched.update_state_after_alloc(
        request, _BlocksMock(([1, 2],)), num_external_tokens=32
    )
    assert sched._push_pending_registrations["req-d"]["remote_dcp_size"] == 8


def test_decode_registration_handshakes_every_producer_shard():
    w = object.__new__(NixlPushConnectorWorker)
    w.shutdown = lambda: None
    w._ensure_handshake = MagicMock(return_value=None)
    w._do_send_reg_notif = MagicMock()
    w._send_registration_to_p(
        "req-d",
        {
            "remote_engine_id": "P",
            "remote_host": "h",
            "remote_port": 1,
            "remote_tp_size": 8,
            "remote_dcp_size": 8,
        },
    )
    assert w._ensure_handshake.call_args.kwargs["dcp_size"] == 8


def _gpu_only_dcp_shard(dcp_rank, dcp_size=4):
    """DCP shard pushing into a GPU-only decode with P's group layout: no
    mixed memory, so only DCP forces the token-position (region) path."""
    w = _dcp_shard(dcp_rank, dcp_size)
    w._dst_desc_pos, w._dst_desc_is_dram = {}, {}
    w._dst_dram_xfer_side_handles = defaultdict(dict)
    w.dst_region_mem_types = {D: ["VRAM", "VRAM"]}
    w.dst_xfer_side_handles = {D: {0: 900, 1: 901}}
    return w


def test_gpu_only_dcp_shard_pairs_pages_by_token_position():
    w = _gpu_only_dcp_shard(dcp_rank=1)
    num_pages = 6
    meta = _meta(
        [_shard_pages(1, 4, num_pages, 50), _shard_pages(1, 4, num_pages, 70)],
        [[10 + g for g in range(num_pages)], [30 + g for g in range(num_pages)]],
        num_tokens=num_pages * BLOCK_SIZE,
        num_computed=(0, 0),
    )
    w._xfer_blocks_for_req("p-req", meta)
    # Shard 1 owns global pages 1 and 5 in both regions; one WRITE per D rank.
    for call in w.nixl_wrapper.make_prepped_xfer.call_args_list:
        assert list(call.args[2]) == [51, 55, 171, 175]
        assert list(call.args[4]) == [11, 15, 131, 135]
    assert [c.args[3] for c in w.nixl_wrapper.make_prepped_xfer.call_args_list] == [
        900,
        901,
    ]


def test_gpu_only_dcp_shard_with_no_pages_notifies_and_reports_done():
    w = _gpu_only_dcp_shard(dcp_rank=3)
    meta = _meta(
        [[], []], [[10, 11], [30, 31]], num_tokens=2 * BLOCK_SIZE, num_computed=(0, 0)
    )
    w._xfer_blocks_for_req("p-req", meta)
    w.nixl_wrapper.make_prepped_xfer.assert_not_called()
    assert w.nixl_wrapper.send_notif.call_count == 2
    assert w._sending_transfers["p-req"] == []


@pytest.mark.parametrize("p_finishes_first", [True, False])
def test_duplicate_registrations_from_every_decode_rank_start_one_push(
    p_finishes_first,
):
    """All D TP ranks send the same PUSH_REG to each P shard. The first one
    starts the push (it WRITEs every D rank); the rest must be dropped, not
    parked forever, and the bookkeeping clears once the push finishes."""
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        PUSH_REG_NOTIF_PREFIX,
    )

    from .test_nixl_push_connector import _StubWriterWorker

    w = _StubWriterWorker.fresh()
    reg = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(
        {"request_id": "req-abc-9999ffff", "local_block_ids": [[5, 6]]}
    )
    if p_finishes_first:
        w._push_finished_blocks["req-abc-1234abcd"] = ([1, 2],)
        for _ in range(8):
            w._handle_push_reg_notif(reg)
    else:
        w._handle_push_reg_notif(reg)
        w._finished_blocks_inbox.put(("req-abc-1234abcd", ([1, 2],)))
        w._push_writer_stop.set()  # run the loop body exactly once
        w._push_writer_stop.clear()
        rid, blocks = w._finished_blocks_inbox.get_nowait()
        matched = w._pop_matching_registration(rid)
        w._mark_push_started(rid, matched["request_id"])
        w._do_start_push_kv(rid, blocks, matched)
        for _ in range(7):
            w._handle_push_reg_notif(reg)

    assert len(w.start_push_calls) == 1
    assert w._pending_d_registrations == {}
    assert w._started_push_reg_ids == {"req-abc-9999ffff": "req-abc-1234abcd"}
