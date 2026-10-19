# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

A dedicated ``nixl-push-writer`` thread owns all push-related NIXL ops:
calls ``get_new_notifs`` (routing PUSH_REG internally; HB / completion
notifs are forwarded to the engine main thread), sends PUSH_REG via
``send_notif``, matches D registrations with P finished blocks, and
issues WRITE transfers via ``make_prepped_xfer`` / ``transfer``.

The engine main thread feeds the writer through queues:
``_reg_send_inbox`` (D-side regs to send), ``_finished_blocks_inbox``
(P-side blocks from metadata) and ``_pending_completion_notifs``
(non-PUSH_REG notifs forwarded back for HB / completion accounting).
The handshake-completion callback feeds ``_deferred_push_inbox`` with
matched pushes whose P→D handshake has finished so the writer can
(re-)issue the WRITE without ever blocking on the network.

Wake model: the writer self-polls every
``_PUSH_WRITER_POLL_INTERVAL_MS`` only while it has unmatched
``_push_finished_blocks`` (i.e. P-side blocks waiting for a D PUSH_REG
notif that has no other wake source). All other progress is
event-driven: the engine main thread sets ``_push_writer_wake`` from
``start_load_kv`` (when handing it new work) and from ``get_finished``
(so each engine step gives the writer a chance to drain NIXL notifs);
the handshake-completion callback sets the same event after a deferred
PUSH_REG send or a deferred push WRITE has been queued. When a request's
lease expires (the base worker reports it via ``done_sending``) or the WRITE completes,
``get_finished`` enqueues an eviction onto ``_evict_finished_inbox`` so
the writer drops any leftover ``_push_finished_blocks`` /
``_pending_d_registrations`` and stops self-polling.
"""

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlConnectorMetadata,
    RemoteMeta,
    ReqId,
    ReqMeta,
    TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import get_base_request_id
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Writer-thread poll cadence while there is in-flight push state. When
# fully idle, the writer blocks on a wake event signalled by the engine
# main thread (start_load_kv / get_finished). Smaller -> lower latency
# while active, slightly more CPU.
_PUSH_WRITER_POLL_INTERVAL_MS = 1.0


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic. See module docstring."""

    # Distinguishes push from pull in the NIXL compatibility hash.
    _TRANSFER_MODE: str = "push"

    _supports_pp_hma = True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Heartbeat handshakes to a PP-sharded producer must be notif-only,
        # like the PUSH_REG path.
        self._hb_handshake_notif_only = True

        # Push-specific state.
        # P-side: outgoing WRITE handles awaiting completion, keyed by
        # request_id. Mutated by writer (submit) and main thread
        # (``_pop_done_transfers``); guarded by
        # ``_sending_transfers_lock``.
        self._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        self._send_failures: set[ReqId] = set()
        self._sending_transfers_lock = threading.Lock()
        # Completion notifs of WRITEs split across a mixed-memory consumer's
        # DRAM and device dlists, sent once all of the request's WRITEs are
        # done. Guarded by ``_sending_transfers_lock``.
        self._pending_push_notifs: dict[ReqId, list[tuple[str, bytes]]] = {}
        # Writer-thread scratch: notifs staged while one request's WRITEs post.
        self._pending_push_notifs_staged: list[tuple[str, bytes]] = []

        # Writer-thread owned matching state.
        # P-side: finished request blocks received from scheduler metadata
        # that have not yet been matched with an incoming D registration.
        self._push_finished_blocks: dict[ReqId, BlockIds] = {}
        # P-side: D registrations received via NIXL notification that have
        # not yet been matched with a finished P request.
        self._pending_d_registrations: dict[ReqId, dict[str, Any]] = {}
        # P-side: consumer request id -> producer request id of pushes already
        # started, so duplicate registrations (one per consumer TP rank) are
        # dropped instead of parked. Cleared when the push finishes.
        self._started_push_reg_ids: dict[ReqId, ReqId] = {}

        # Cross-thread channels.
        self._reg_send_inbox: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._finished_blocks_inbox: queue.Queue[tuple[str, BlockIds]] = queue.Queue()
        self._pending_completion_notifs: queue.Queue[bytes] = queue.Queue()
        # Main thread → writer: req_ids whose lease has expired or whose
        # WRITE has completed. Writer drops them from
        # ``_push_finished_blocks`` so an unmatched entry doesn't keep the
        # writer busy-polling forever.
        self._evict_finished_inbox: queue.Queue[str] = queue.Queue()
        # Handshakes that have just completed and are ready for the WRITE on wthread
        self._deferred_push_inbox = queue.Queue[tuple[str, BlockIds, dict[str, Any]]]()

        # Wake signal from engine main thread (start_load_kv / get_finished).
        # Writer self-polls at _PUSH_WRITER_POLL_INTERVAL_MS while it has
        # active in-flight state; otherwise it blocks until signalled.
        self._push_writer_wake = threading.Event()

        self._push_writer_stop = threading.Event()
        self._push_writer_thread: threading.Thread | None = None

    # --- Lifecycle ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        super().register_kv_caches(kv_caches)
        # A mixed-memory consumer (HiSparse decode) only receives WRITEs: the
        # producer splits each WRITE across the consumer's DRAM and device
        # dlists. A mixed-memory producer would also need per-type local
        # dlists on the WRITE path.
        if self._mixed_mem_types and self.kv_transfer_config.is_kv_producer:
            raise NotImplementedError(
                "NixlPushConnector does not support a mixed-memory producer"
            )
        if self._push_writer_thread is None:
            self._push_writer_thread = threading.Thread(
                target=self._push_writer_loop,
                daemon=True,
                name="nixl-push-writer",
            )
            self._push_writer_thread.start()
            logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def shutdown(self) -> None:
        self._push_writer_stop.set()
        # Unblock the writer if it's waiting in the no-active-state branch.
        self._push_writer_wake.set()
        if self._push_writer_thread is not None:
            self._push_writer_thread.join(timeout=2)
            self._push_writer_thread = None
        with self._sending_transfers_lock:
            for handles in self._sending_transfers.values():
                for handle in handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
            self._sending_transfers.clear()
            self._send_failures.clear()
            self._pending_push_notifs.clear()
        super().shutdown()

    # --- Engine-main-thread entry point -------------------------------- #

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Pre-process metadata; defer NIXL ops to the writer thread."""
        if self.pcp_rank > 0 and not self.pcp_dcp_sharded:
            return

        # D-side: track reqs waiting for P to push.
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            remote_engine_id = meta.remote.engine_id
            # Update last activity from this remote (same as pull), but only for
            # an already-connected engine so we never leave an _engine_last_active
            # entry without a _remote_agents entry (breaks _cleanup_remote_engine).
            if remote_engine_id in self._remote_agents:
                self._engine_last_active[remote_engine_id] = time.perf_counter()

            logger.debug(
                "start_load_kv (push) for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            if self._mixed_mem_types:
                meta.region_blocks_to_zero = self._unwritten_region_blocks(meta)
            self._recving_metadata[req_id] = meta

        # --- D-side: registrations to send to P via NIXL ---
        if metadata.push_registrations:
            for req_id, reg_data in metadata.push_registrations.items():
                self._reg_send_inbox.put((req_id, reg_data))
            self._push_writer_wake.set()

        # --- P-side: newly finished blocks awaiting a D registration match ---
        if metadata.push_finished_blocks:
            for req_id, block_ids in metadata.push_finished_blocks.items():
                self._finished_blocks_inbox.put((req_id, block_ids))
            self._push_writer_wake.set()

        # Batch + lease tracking (same as pull).
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            assert req_id not in self._reqs_to_send
        # Rebase scheduler-clock deadlines onto this worker's clock — see the
        # equivalent block in pull_worker.start_load_kv for the rationale.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        # Heartbeats still leave from the main thread (base worker behaviour).
        self._send_heartbeats(metadata)

    def _handshake_target_ranks(
        self, remote_tp_size: int, remote_dcp_size: int
    ) -> list[int]:
        """A DCP-sharded producer pushes its token slice to every consumer rank.

        The read-side choice (one remote rank per local DCP slice) suits a
        consumer that pulls; a pushing shard must instead reach each copy of
        the consumer's replicated MLA cache itself.
        """
        if (
            self.dcp_size > 1
            and remote_dcp_size == 1
            and self.kv_transfer_config.is_kv_producer
        ):
            return list(range(remote_tp_size))
        return super()._handshake_target_ranks(remote_tp_size, remote_dcp_size)

    def _unwritten_region_blocks(self, meta: ReqMeta) -> BlockIds:
        """Physical pages of a mixed-memory consumer that no WRITE will fill.

        A producer pushes into this consumer region by region, covering only
        the request's token range; pages past it are allocation padding, so
        they are zeroed on completion (as the pull worker does after a
        region-mapped read).
        """
        regions = self._block_ids_by_region(
            meta.local_physical_block_ids, self.region_group_ids
        )
        num_tokens = meta.remote.num_tokens if meta.remote is not None else None
        if num_tokens is None:
            return [[] for _ in regions]
        num_pages = cdiv(num_tokens, self.block_size)
        transfer_groups = self.kv_cache_config.transfer_group_ids
        to_zero: list[list[int]] = []
        for region, group in zip(regions, self.region_group_ids, strict=True):
            cached = (
                meta.local_num_computed_blocks[transfer_groups[group]]
                * self._physical_blocks_per_logical_kv_block
                if group >= 0 and meta.local_num_computed_blocks
                else 0
            )
            to_zero.append(list(region[max(num_pages - cached, 0) :]))
        return to_zero

    # --- Writer thread ------------------------------------------------- #

    def _push_writer_loop(self) -> None:
        sleep_s = _PUSH_WRITER_POLL_INTERVAL_MS / 1000.0

        while not self._push_writer_stop.is_set():
            try:
                # 1. D registrations to send.
                while True:
                    try:
                        rid, rd = self._reg_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_registration_to_p(rid, rd)

                # 2. Deferred P→D pushes whose handshake just completed; do xfer now
                while True:
                    try:
                        rid, blocks, rd = self._deferred_push_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._do_start_push_kv(rid, blocks, rd)

                # 3. P-side finished blocks; match against pending regs.
                while True:
                    try:
                        rid, blocks = self._finished_blocks_inbox.get_nowait()
                    except queue.Empty:
                        break
                    matched = self._pop_matching_registration(rid)
                    if matched is not None:
                        self._mark_push_started(rid, matched["request_id"])
                        self._do_start_push_kv(rid, blocks, matched)
                    else:
                        self._push_finished_blocks[rid] = blocks

                # 3b. Evict finished blocks for requests that have either
                # completed (WRITE acknowledged) or whose lease expired
                # without a D registration.  Drop pending registrations
                # for the same reason so we don't leak state.
                while True:
                    try:
                        rid = self._evict_finished_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._push_finished_blocks.pop(rid, None)
                    self._pending_d_registrations.pop(rid, None)
                    # Registrations are keyed by the consumer's request id.
                    for reg_id, req_id in list(self._started_push_reg_ids.items()):
                        if req_id == rid:
                            del self._started_push_reg_ids[reg_id]
                            self._pending_d_registrations.pop(reg_id, None)

                # 4. NIXL notifs: route PUSH_REG; forward the rest.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            # Self-poll only while there is no other wake source: P-side
            # finished blocks waiting for a D PUSH_REG match. All other
            # progress is event-driven (see module docstring).
            if self._push_finished_blocks:
                self._push_writer_stop.wait(timeout=sleep_s)
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        try:
            reg_data = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        except Exception:
            logger.exception("Failed to decode PUSH_REG notification payload")
            return
        rid = reg_data.get("request_id") if isinstance(reg_data, dict) else None
        if not isinstance(rid, str):
            logger.warning("PUSH_REG notif missing request_id; dropping")
            return

        if rid in self._started_push_reg_ids:
            # Every consumer TP rank sends the same registration; the first one
            # already started this push (it WRITEs every consumer rank).
            return
        match = self._pop_matching_finished_blocks(rid)
        if match is not None:
            fin_id, blocks = match
            self._mark_push_started(fin_id, rid)
            self._do_start_push_kv(fin_id, blocks, reg_data)
        else:
            self._pending_d_registrations[rid] = reg_data

    def _mark_push_started(self, request_id: str, registration_id: str) -> None:
        """Drop duplicate registrations of a push that has just started."""
        self._started_push_reg_ids[registration_id] = request_id
        self._pending_d_registrations.pop(registration_id, None)

    # --- D-side registration send (writer thread) ---------------------- #

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        """Handshake (if needed) then send PUSH_REG. ``send_notif`` always
        executes on the writer; the handshake runs on the background executor
        and the request is re-queued onto ``_reg_send_inbox`` once it
        completes (at which point ``_ensure_handshake`` returns ``None`` and we
        send directly)."""
        remote_pp_size = reg_data.get("remote_pp_size", 1)
        fut = self._ensure_handshake(
            reg_data["remote_engine_id"],
            reg_data["remote_host"],
            reg_data["remote_port"],
            reg_data["remote_tp_size"],
            # A DCP-sharded producer holds one token slice per rank, so every
            # rank must receive the registration to push its slice.
            dcp_size=reg_data.get("remote_dcp_size", 1),
            pp_size=remote_pp_size,
            # D only ever sends PUSH_REG notifs to P and never reads or writes
            # P's memory in push mode, so it never needs the transfer
            # descriptors set up by the full add_remote_agent path.
            notif_agents_only=True,
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
            return

        def _on_handshake(
            f: Future[tuple[dict[tuple[int, int], str], float]],
            rid: str = req_id,
            rd: dict[str, Any] = reg_data,
        ) -> None:
            try:
                f.result()
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_handshake_failed", req_id=rid, error=e
                )
                self._failed_recv_reqs.put(rid)
                return
            # Re-queue for the writer to send now that the handshake is done.
            self._reg_send_inbox.put((rid, rd))
            # Wake the writer so it sends the PUSH_REG promptly even if
            # otherwise parked.
            self._push_writer_wake.set()

        fut.add_done_callback(_on_handshake)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        engine_id = reg_data["remote_engine_id"]
        notif_msg = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg_data)
        # _remote_agents is mutated on other threads; snapshot under the lock.
        with self._handshake_lock:
            agents = dict(self._remote_agents.get(engine_id) or {})
        if not agents:
            logger.error(
                "No remote agents for engine %s; cannot send registration for %s",
                engine_id,
                req_id,
            )
            self.xfer_stats.record_failed_notification()
            self._failed_recv_reqs.put(req_id)
            return
        for rank, agent_name in agents.items():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_notif_failed",
                    req_id=req_id,
                    error=e,
                    remote_rank=rank,
                )
                self.xfer_stats.record_failed_notification()
                # Earlier registrations may still trigger WRITEs into D's blocks.
                # Keep the receive pending until those writes are finished.
        logger.debug(
            "Sent PUSH_REG for %s to engine %s (%dB)", req_id, engine_id, len(notif_msg)
        )

    # --- Matching helpers --------------------------------------------- #

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        """Pop the D-side registration matching *request_id*.

        Exact key first, then a match after stripping the random suffix from
        both sides. No match leaves the request unmatched (push not started).
        """
        data = self._pending_d_registrations.pop(request_id, None)
        if data is not None:
            return data
        base_id = get_base_request_id(request_id)
        for reg_id in list(self._pending_d_registrations):
            if get_base_request_id(reg_id) == base_id:
                return self._pending_d_registrations.pop(reg_id)
        return None

    def _pop_matching_finished_blocks(
        self, request_id: str
    ) -> tuple[str, BlockIds] | None:
        """Pop the P-side finished blocks matching *request_id*.

        Same lookup as ``_pop_matching_registration``: exact key, then a
        match after stripping the random suffix from both sides.
        """
        blocks = self._push_finished_blocks.pop(request_id, None)
        if blocks is not None:
            return request_id, blocks
        base_id = get_base_request_id(request_id)
        for fin_id in list(self._push_finished_blocks):
            if get_base_request_id(fin_id) == base_id:
                return fin_id, self._push_finished_blocks.pop(fin_id)
        return None

    # --- WRITE transfer logic (writer thread) ------------------------- #

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Start push-based KV transfer from P worker to D node.

        The P→D handshake runs on the base worker's background executor.
        If it isn't ready yet we register a completion callback, defer the
        WRITE, and re-drive this request via ``_deferred_push_inbox`` once
        the handshake resolves -- so the writer thread never blocks on the
        network (mirrors ``_send_registration_to_p``).
        """
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            return

        # ``local_block_ids`` are P's logical block IDs; ``remote_block_ids``
        # (D's, from the PUSH_REG notif) are also logical.
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_request_id = registration_data["request_id"]

        # Runs on the background executor; defer the WRITE until it's ready.
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            registration_data["decode_port"],
            registration_data["decode_tp_size"],
        )
        if fut is not None:

            def _on_handshake(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                rid: str = request_id,
                blocks: BlockIds = local_block_ids,
                rd: dict[str, Any] = registration_data,
            ) -> None:
                if (e := f.exception()) is not None:
                    # The engine reclaims the blocks via the TTL so we dont free here
                    self._log_failure(
                        failure_type="push_handshake_failed", req_id=rid, error=e
                    )
                    return
                self._deferred_push_inbox.put((rid, blocks, rd))
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return
        # Keep the engine alive while it is actively receiving pushes, mirroring
        # how pull-mode transfers touch _engine_last_active in start_load_kv.
        self._engine_last_active[decode_engine_id] = time.perf_counter()

        logical_local = self._as_grouped_block_ids(local_block_ids)
        logical_remote = self._as_grouped_block_ids(remote_block_ids)
        physical_local = self._logical_to_kernel_block_ids(
            logical_local, self._physical_blocks_per_logical_kv_block
        )

        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=logical_remote,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
                # Absent from registrations sent by older consumers.
                num_tokens=registration_data.get("num_tokens"),
                num_computed_blocks=tuple(
                    registration_data.get("num_computed_blocks") or ()
                ),
            ),
        )

        t0 = time.perf_counter()
        self._xfer_blocks_for_req(req_id=request_id, meta=push_meta)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 200.0:
            logger.warning(
                "_do_start_push_kv for %s took %.1fms (slow NIXL submission)",
                request_id,
                elapsed_ms,
            )

    @staticmethod
    def _as_grouped_block_ids(block_ids: BlockIds) -> BlockIds:
        """Normalise a sequence of block IDs to a tuple-of-groups shape.

        ``BlockIds`` is canonically a tuple of per-group lists, but some
        registration payloads collapse a single-group case to a flat
        list. Re-wrap that case so downstream group-aware helpers see a
        consistent shape."""
        if block_ids and not isinstance(block_ids[0], (list, tuple)):
            return (list(block_ids),)
        return block_ids

    def _xfer_blocks_for_req(self, req_id: str, meta: ReqMeta):
        """Issue WRITE transfers to one or more remote TP ranks."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        # Expand D's logical IDs using the ratio learned during the
        # NIXL handshake. ``meta`` is freshly built by
        # ``_do_start_push_kv`` so mutating it here is safe.
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        local_region_groups = self.region_group_ids
        remote_region_groups = self.dst_region_group_ids[engine_id]
        groups_differ = local_region_groups != remote_region_groups
        mixed_remote = engine_id in self._dst_desc_pos
        # A DCP-sharded producer holds every dcp_size-th page, so it pairs pages
        # by token position, which the region path does.
        if (
            (groups_differ and not self._transfer_layer_group_ids)
            or mixed_remote
            or self.dcp_size > 1
        ):
            if not self.use_mla or self._has_mamba:
                raise NotImplementedError(
                    "NixlPushConnector supports different producer and consumer "
                    "cache-group layouts for pure MLA models only"
                )
            self._xfer_blocks_by_region(
                req_id, meta, local_region_groups, remote_region_groups, mixed_remote
            )
            return

        # MLA latent is replicated across D's TP ranks: the tp-mapping
        # collapses it to one rank (fine for reads), but push must WRITE every
        # D rank or the rest decode stale KV. For hybrid MLA+SSM the sharded
        # SSM state already targets every covered D rank, so only the
        # attention groups need widening; pure MLA writes to all handshaked
        # ranks (only the dst differs per rank).
        replicate_attn = self.use_mla and tp_ratio < 0
        if replicate_attn and not self._has_mamba:
            assert len(plan.all_source_ranks) == 1
            write_ranks = sorted(self.dst_xfer_side_handles[engine_id])
        else:
            write_ranks = list(plan.all_source_ranks)

        num_groups = len(local_block_ids)

        def group_ids(block_ids: BlockIds, rank: int) -> BlockIds:
            return [
                list(block_ids[g])
                if (self._is_csa_linear and tp_ratio < 0)
                or (replicate_attn and _is_attention_spec(self._group_spec_types[g]))
                or rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ]

        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=group_ids(local_block_ids, rank),
                remote_block_ids=group_ids(remote_block_ids, rank),
            )
            for rank in write_ranks
        ]

        handles: list[int] = []
        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _xfer_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            if tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1):
                # Multiple targets: write each rank its chunk of local memory.
                # Hybrid MLA+SSM also lands here: its split handles replicate
                # the attention descriptors and chunk only the SSM state.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )
            if handle is not None:
                handles.append(handle)

        # Publish all the request's WRITE handles in one locked update: a
        # partial set would let ``_pop_done_transfers`` finish the request
        # early, then double-report it as the remaining writes land.
        if handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)

    def _xfer_blocks_by_region(
        self,
        req_id: str,
        meta: ReqMeta,
        local_region_groups: list[int],
        remote_region_groups: list[int],
        mixed_remote: bool,
    ) -> None:
        """WRITE MLA KV region by region into a consumer with another layout.

        A HiSparse decode keeps the sparse-MLA cache in a host-resident group
        and the indexer cache in a device group, while the producer keeps both
        in device groups. Region ``i`` holds the same layer on both sides
        (checked at handshake), so pair the blocks per region rather than per
        cache group. Mirrors the region-mapped path of the pull worker.
        """
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        if self.block_size != remote_info.remote_block_size:
            raise NotImplementedError(
                "Region-mapped NIXL transfers require matching physical block sizes"
            )
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)
        local_by_region = self._block_ids_by_region(
            meta.local_physical_block_ids, local_region_groups
        )
        remote_by_region = self._block_ids_by_region(
            meta.remote.block_ids, remote_region_groups
        )
        remote_by_region, local_by_region = self._pair_region_pages(
            meta.remote,
            remote_info,
            remote_region_groups,
            remote_by_region,
            local_by_region,
        )

        # MLA is replicated over D's TP ranks: every handshaken rank gets a
        # copy (see _xfer_blocks_for_req). Each DCP shard holds a distinct token
        # slice, so it too writes every consumer rank.
        if tp_ratio < 0 or self.dcp_size > 1:
            assert len(plan.all_source_ranks) == 1
            write_ranks = sorted(self.dst_xfer_side_handles[engine_id])
        else:
            write_ranks = list(plan.all_source_ranks)
        local_xfer_side_handle = self.src_xfer_handles_by_block_size[self.block_size]

        if not any(local_by_region):
            # None of the request's uncached pages fall in this rank's DCP
            # slice (short prompt, or a long prefix hit on D). Each consumer
            # rank still counts one notif per shard, and this rank must still
            # report finished_sending so P frees its blocks.
            notif_id = f"{meta.remote.request_id}:{self.transfer_tp_size}".encode()
            self._send_push_notifs(
                req_id,
                [
                    (self._remote_agents[engine_id][(0, rank)], notif_id)
                    for rank in write_ranks
                ],
            )
            with self._sending_transfers_lock:
                self._sending_transfers.setdefault(req_id, [])
            return

        handles: list[int] = []
        for rank in write_ranks:
            spec = ReadSpec(rank, local_by_region, remote_by_region, True)
            if mixed_remote:
                handles.extend(
                    self._xfer_blocks_mixed(
                        read_spec=spec,
                        dst_engine_id=engine_id,
                        request_id=req_id,
                        remote_request_id=meta.remote.request_id,
                        local_xfer_side_handle=local_xfer_side_handle,
                    )
                )
                continue
            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=self.dst_xfer_side_handles[engine_id][rank],
            )
            if handle is not None:
                handles.append(handle)

        staged_notifs = self._pending_push_notifs_staged
        self._pending_push_notifs_staged = []
        if handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)
                if staged_notifs:
                    self._pending_push_notifs.setdefault(req_id, []).extend(
                        staged_notifs
                    )

    def _pair_region_pages(
        self,
        remote: RemoteMeta,
        remote_info: Any,
        remote_region_groups: list[int],
        remote_by_region: BlockIds,
        local_by_region: BlockIds,
    ) -> tuple[BlockIds, BlockIds]:
        """Pair the consumer's uncached pages with this rank's, per region.

        Returns ``(consumer pages, local pages)`` of equal length per region.
        The consumer registered only pages past its prefix-cache hit, and
        either list may end in allocation padding, so pages are paired by
        token position. A DCP-sharded producer holds global page ``g`` at
        local index ``g // dcp_size`` when ``g % dcp_size == dcp_rank``.
        """
        remote_groups = remote.num_computed_blocks
        if not (
            remote.num_tokens is not None
            and remote_groups
            and all(group >= 0 for group in remote_region_groups)
        ):
            if self.dcp_size > 1 or self._physical_blocks_per_logical_kv_block != (
                remote_info.remote_physical_blocks_per_logical
            ):
                raise NotImplementedError(
                    "Region-mapped pushes with DCP or different logical block "
                    "sizes need the consumer's token count and per-group prefix "
                    "counts"
                )
            return self._apply_prefix_caching_by_region(
                remote_by_region, local_by_region
            )
        if self.dcp_size > 1 and (
            self._physical_blocks_per_logical_kv_block != 1
            or remote_info.remote_physical_blocks_per_logical != 1
        ):
            raise NotImplementedError(
                "DCP region pushes require matching logical and physical block sizes"
            )

        num_pages = cdiv(remote.num_tokens, self.block_size)
        paired_remote: list[list[int]] = []
        paired_local: list[list[int]] = []
        for remote_pages, local_pages, group in zip(
            remote_by_region, local_by_region, remote_region_groups, strict=True
        ):
            cached = (
                remote_groups[group] * remote_info.remote_physical_blocks_per_logical
            )
            # The consumer is never DCP-sharded (see NixlPushConnector), so it
            # takes the unsharded side of the pull-direction matcher.
            consumer, producer = self._apply_dcp_prefix_caching(
                list(remote_pages),
                list(local_pages),
                remote_rank=self.dcp_rank,
                local_dcp_size=1,
                local_dcp_rank=0,
                remote_dcp_size=self.dcp_size,
                local_num_computed_blocks=cached,
                num_remote_blocks=num_pages,
            )
            paired_remote.append(consumer)
            paired_local.append(producer)
        return paired_remote, paired_local

    def _write_desc_ids(
        self, read_spec: ReadSpec, dst_engine_id: str, request_id: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Local and remote descriptor ids for one WRITE, or ``None`` if the
        spec carries no block groups at all."""
        assert self.transfer_topo is not None
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            if read_spec.block_ids_by_region:
                raise NotImplementedError(
                    "Region-mapped NIXL transfers require matching physical block sizes"
                )
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return None

        if read_spec.block_ids_by_region:
            # Lists are per region and already prefix-trimmed by
            # _xfer_blocks_for_req; region i of both sides holds the same layer.
            region_group_ids: list[int] | None = list(range(len(local_block_ids)))
            remote_region_group_ids = region_group_ids
            uses_region_group_mapping = remote_uses_region_group_mapping = (
                len(local_block_ids) > 1
            )
        else:
            # Prefix caching: D allocated only uncached blocks, so on a partial
            # hit it sends fewer than P's. End-trim P's blocks to that same
            # suffix so we WRITE only the uncomputed tail into D's slots. Runs
            # on kernel ids, post-expansion.
            remote_block_ids, local_block_ids = self._apply_prefix_caching(
                decode_block_ids=remote_block_ids,
                prefill_block_ids=local_block_ids,
                decode_physical_per_logical=(
                    remote_info.remote_physical_blocks_per_logical
                ),
                prefill_physical_per_logical=self._physical_blocks_per_logical_kv_block,
            )
            region_group_ids = self.region_group_ids
            uses_region_group_mapping = self._uses_region_group_mapping
            remote_region_group_ids = self.dst_region_group_ids[dst_engine_id]
            remote_uses_region_group_mapping = self.dst_uses_region_group_mapping[
                dst_engine_id
            ]

        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        assert len(local_block_ids) == len(remote_block_ids), (
            f"push group-count mismatch for {request_id}: {len(local_block_ids)} "
            f"local vs {len(remote_block_ids)} remote groups"
        )
        for i in range(len(local_block_ids)):
            assert len(local_block_ids[i]) == len(remote_block_ids[i]), (
                f"push block-count mismatch for {request_id} group {i}: "
                f"{len(local_block_ids[i])} local vs "
                f"{len(remote_block_ids[i])} remote blocks"
            )

        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
            region_num_blocks=self.dst_region_num_blocks[dst_engine_id],
            region_group_ids=remote_region_group_ids,
            uses_region_group_mapping=remote_uses_region_group_mapping,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=self.dst_region_num_blocks[self.engine_id],
            region_group_ids=region_group_ids,
            uses_region_group_mapping=uses_region_group_mapping,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)
        return np.asarray(local_block_descs_ids), np.asarray(remote_block_descs_ids)

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ) -> int | None:
        """Post a WRITE point-to-point xfer request.

        Returns the in-flight transfer handle (so the caller can track all of
        a request's handles atomically), or ``None`` if nothing was submitted.
        """
        remote_rank = read_spec.remote_rank
        notif_id = f"{remote_request_id}:{self.transfer_tp_size}".encode()
        desc_ids = self._write_desc_ids(read_spec, dst_engine_id, request_id)
        if desc_ids is None:
            return None
        local_block_descs_ids, remote_block_descs_ids = desc_ids

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            # Caller tracks the handle (atomically with the request's other
            # writes) so P can free blocks once all of them are done.
            return handle
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Push WRITE submission failed; releasing handle",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            # On the P side this WRITE failure is purely outbound; we
            # don't have a ``_recving_metadata`` entry to invalidate, so
            # we just release the handle and let the engine reschedule
            # via the lease / watchdog.
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            if not self._handle_failed_transfer(request_id, handle):
                return handle
            return None

    def _xfer_blocks_mixed(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
    ) -> list[int]:
        """Split a WRITE across a mixed-memory consumer's DRAM and device dlists.

        Two WRITEs may complete in either order, so neither carries the
        completion notif; ``get_transfer_results`` sends it once every WRITE of
        the request is done. A remote rank without a DRAM handle shares its
        host pool with TP rank 0, whose WRITE delivers that part.

        Returns the posted handles; the caller publishes them with the
        notif recorded in ``_pending_push_notifs``.
        """
        remote_rank = read_spec.remote_rank
        agent_name = self._remote_agents[dst_engine_id][(0, remote_rank)]
        notif = (
            agent_name,
            f"{remote_request_id}:{self.transfer_tp_size}".encode(),
        )
        desc_ids = self._write_desc_ids(read_spec, dst_engine_id, request_id)
        if desc_ids is None:
            return []
        local_ids, remote_ids = desc_ids
        is_dram = self._dst_desc_is_dram[dst_engine_id][remote_ids]
        remote_pos = self._dst_desc_pos[dst_engine_id][remote_ids]
        dram_handle = self._dst_dram_xfer_side_handles[dst_engine_id].get(remote_rank)
        if dram_handle is None:
            keep = ~is_dram
            local_ids, remote_pos, is_dram = (
                local_ids[keep],
                remote_pos[keep],
                is_dram[keep],
            )
        writes = (
            (is_dram, dram_handle),
            (~is_dram, self.dst_xfer_side_handles[dst_engine_id][remote_rank]),
        )

        handles: list[int] = []
        try:
            for mask, remote_handle in writes:
                if not mask.any():
                    continue
                handle = self.nixl_wrapper.make_prepped_xfer(
                    "WRITE",
                    local_xfer_side_handle,
                    local_ids[mask],
                    remote_handle,
                    remote_pos[mask],
                )
                handles.append(handle)
                self.nixl_wrapper.transfer(handle)
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Push WRITE submission failed; releasing handles",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            # A posted WRITE may be in flight; keep any handle that will not
            # release so P holds its blocks until the DMA is done.
            return [
                handle
                for handle in handles
                if not self._handle_failed_transfer(request_id, handle)
            ]

        if handles:
            self._pending_push_notifs_staged.append(notif)
        else:
            # Nothing left to write to this rank (its part is in rank 0's pool).
            self._send_push_notifs(request_id, [notif])
        return handles

    def _send_push_notifs(
        self, request_id: str, notifs: list[tuple[str, bytes]]
    ) -> None:
        """Tell D ranks that every WRITE of a mixed-memory push has landed."""
        for agent_name, notif_id in notifs:
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="D waits for this push until its registration times out.",
                    req_id=request_id,
                    error=e,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()

    # --- Notification handling on engine main thread ------------------ #

    def _get_new_notifs(self) -> set[str]:
        """Drain HB / completion notifs forwarded by the writer thread.

        The writer owns ``nixl_wrapper.get_new_notifs`` for push; PUSH_REG
        notifs are handled there. Everything else is forwarded here for
        existing accounting.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        while True:
            try:
                notif = self._pending_completion_notifs.get_nowait()
            except queue.Empty:
                break

            msg = notif.decode("utf-8")
            if msg.startswith("HB:"):
                self._handle_heartbeat(msg[3:])
                continue

            req_id, tp_size = msg.rsplit(":", 1)

            # Not tracked as a P-side send/process for this notif.
            if req_id not in self._reqs_to_send and req_id not in self._reqs_to_process:
                if (meta := self._recving_metadata.get(req_id)) is not None:
                    # Consumer waits for one notif per producer rank writing
                    # here: pp_size stages * producers-per-consumer (>1 when
                    # producer TP > consumer TP, and for a DCP-sharded producer,
                    # each of whose ranks writes its own token slice; tp_size is
                    # the producer's transfer TP).
                    producers_per_consumer = max(
                        1, int(tp_size) // self.transfer_tp_size
                    )
                    if meta.dcp_size > 1:
                        # Every shard writes to every (unsharded) consumer rank.
                        producers_per_consumer = int(tp_size)
                    expected_notifs = meta.pp_size * producers_per_consumer
                    self.consumer_notification_counts_by_req[req_id] += 1
                    notifs = self.consumer_notification_counts_by_req[req_id]
                    if notifs < expected_notifs:
                        continue
                    del self.consumer_notification_counts_by_req[req_id]
                    # P drove the transfer (we own no NIXL handle), so
                    # materialise an empty ``_recving_transfers`` entry for
                    # ``_pop_done_transfers`` to report done.
                    self._recving_transfers.setdefault(req_id, [])
                else:
                    # Not tracked on either side (lease may have expired
                    # before the notif arrived). Log and skip.
                    self.xfer_stats.record_notification_after_expiry()
                    logger.error(
                        "Unrecognized request %s notif (may have expired).",
                        req_id,
                    )
                continue

            n_consumers = int(tp_size)
            tp_ratio = self.transfer_topo.tp_ratio(n_consumers)
            consumers_per_producer = -tp_ratio if n_consumers > self.world_size else 1
            self.consumer_notification_counts_by_req[req_id] += 1
            if (
                self.consumer_notification_counts_by_req[req_id]
                == consumers_per_producer
            ):
                notified_req_ids.add(req_id)
                del self.consumer_notification_counts_by_req[req_id]
                self._reqs_to_process.remove(req_id)
                self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def get_transfer_results(self) -> KVConnectorTransferResults:
        # Engine main thread asking for completions: also wake the writer
        # so it gets a chance to drain NIXL notifs (heartbeats, completion
        # notifs, late PUSH_REGs) even if it had been parked.
        self._push_writer_wake.set()

        results = super().get_transfer_results()
        done_sending = results.finished_sending

        # ``_pop_done_transfers`` mutates ``_sending_transfers``; the
        # writer thread also appends to it, so guard the pop.
        with self._sending_transfers_lock:
            done_pushing, failed_pushing = self._pop_done_transfers(
                self._sending_transfers
            )
            # Remember failures until the final sibling WRITE completes.
            self._send_failures.update(failed_pushing)
            # Split WRITEs into a mixed-memory consumer notify only now that
            # all of them landed. A failed push sends nothing: D then fails the
            # request on its registration watchdog rather than reading partial KV.
            push_notifs = {
                req_id: notifs
                for req_id in done_pushing
                if (notifs := self._pending_push_notifs.pop(req_id, None))
                and req_id not in self._send_failures
            }
            successful = {
                req_id
                for req_id in done_pushing - self._send_failures
                if req_id in self._reqs_to_send or req_id in self._reqs_to_process
            }
            self._send_failures.difference_update(done_pushing | done_sending)
        for req_id, notifs in push_notifs.items():
            self._send_push_notifs(req_id, notifs)
        # Expired requests were already reported, even if their WRITEs finish later.
        done_pushing = successful
        for req_id in done_pushing:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            done_sending.add(req_id)

        # Tell the writer to drop any state it still holds for any
        # request that just finished (push completed) or expired
        # (lease ran out without a D registration ever arriving).
        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return results
