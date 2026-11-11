"""v3 super-step scheduler for workflow.json.

Drives a v3 LangGraph-style workflow. The runtime is a frontier-based
super-step scheduler: each tick, run all ready nodes concurrently, fold
their outputs back into the per-subgraph local-outputs dict, then
advance the frontier by following ``edges`` and resolving
``conditional_edges``. Subgraph nodes recurse into the same scheduler.

Streaming nodes (``streaming: true``) are spawned into a detached
future. They never block downstream readiness; consumers reach them
through ``$ref`` to a ``StreamSlot``. On subgraph teardown the runtime
fires each streaming node's ``CancelToken``, grace-waits, then
force-cancels.

Send (dynamic fan-out) is expressed by a ``router`` node whose function
returns a list of ``{"to", "inputs"}`` dicts; outputs collect into a
list under the router's name.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gap_core.errors import (
    GraphValidationError,
    NodeExecutionError,
    PipelineError,
    TaskCancelled,
    VerificationFailed,
    is_terminal,
)
from gap_core.tools import guards

from .context import CancelToken, NodeContext
from .nodes import (
    execute_script_node,
    execute_tool_node,
)
from .observation_stream import ObservationStream, start_observation_stream
from .tracing import DagTrace
from .validate import validate_workflow
from .workflow import (
    END,
    OBSERVATION_STREAM_INPUT_NAME,
    RESERVED_INPUT_PSEUDOSTATE,
    START,
    NodeDef,
    SubgraphDef,
    Workflow,
    load_workflow,
    resolve_inputs,
    resolve_ref,
)

logger = logging.getLogger(__name__)


def _typed_error_text(exc: BaseException) -> str:
    """``TypeName: message``, because the message alone is often not enough.

    A bare ``KeyError`` stringifies to ``'z'`` and a ``PipelineError`` raised
    by a policy's own check stringifies to the sentence its author wrote. Only
    the type separates "my code indexed a field that is not there" from "my
    check fired" -- and that is the first thing a reader has to decide.
    """
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _node_error_text(exc: BaseException) -> str:
    """The sentence a reader needs: which node, what kind, and what it said.

    ``NodeExecutionError`` carries the node id but renders its cause with
    ``str()``, which drops the cause's type. Rebuild it so the type survives.
    """
    if isinstance(exc, NodeExecutionError):
        return f"Node {exc.node_id!r} failed: {_typed_error_text(exc.cause)}"
    return _typed_error_text(exc)


# ---------------------------------------------------------------------------
# Streaming-node slot
# ---------------------------------------------------------------------------


class StreamSlot:
    """Thread-safe slot holding the latest published value of a streaming node.

    ``$ref`` resolution invokes ``latest()`` to retrieve a snapshot.
    Calls block on ``_first_published`` until the underlying skill
    publishes its first value (or the cancel token fires).
    """

    def __init__(self, name: str, cancel_token: CancelToken) -> None:
        self.name = name
        self._cancel_token = cancel_token
        self._lock = threading.Lock()
        self._value: Any = None
        self._first_published = threading.Event()
        self._closed = False

    def publish(self, value: Any) -> None:
        with self._lock:
            self._value = value
        self._first_published.set()

    def latest(self, timeout_s: float | None = 60.0) -> Any:
        if not self._first_published.is_set():
            ok = self._first_published.wait(timeout=timeout_s)
            if not ok:
                raise PipelineError(
                    f"streaming node {self.name!r} did not publish a "
                    f"value within {timeout_s}s"
                )
        with self._lock:
            return self._value

    def cancel(self) -> None:
        self._cancel_token.cancel()
        # Unblock any consumers waiting on first publish.
        self._first_published.set()
        self._closed = True

    @property
    def cancel_token(self) -> CancelToken:
        return self._cancel_token


# ---------------------------------------------------------------------------
# Per-subgraph execution state
# ---------------------------------------------------------------------------


@dataclass
class _ScopeState:
    """Mutable state for one super-step scheduler invocation.

    A scope is created per subgraph (or top-level workflow). Streaming
    nodes are tracked separately for teardown.
    """

    local_outputs: dict[str, Any] = field(default_factory=dict)
    streaming_slots: dict[str, StreamSlot] = field(default_factory=dict)
    streaming_futures: dict[str, Future] = field(default_factory=dict)
    completed_nodes: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Subgraph-exit observability hook
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgraphExitEvent:
    """Snapshot-time event fired by the executor at every subgraph exit.

    Consumed by harnesses (rehearsal, comparison tools) to take a
    privileged-state World snapshot and evaluate the subgraph's
    checkpoints. Production runs leave ``subgraph_exit_hook=None``; the
    hook then never fires and there is zero overhead.

    ``visit_index`` is 0-based and increments per visit to a given
    subgraph (a loop that visits ``run_sg`` three times produces
    indices ``0, 1, 2``).

    ``error_path=True`` means the inner scope raised and the subgraph
    is exiting through ``on_error``. The snapshot can still be taken
    (we are between the raise and the exit-value bind), but consumers
    should distinguish it from a normal exit.
    """

    sg_name: str
    visit_index: int
    bound_outputs: dict[str, Any]
    exit_value: str
    error_path: bool
    elapsed_s: float


# ---------------------------------------------------------------------------
# Workflow executor
# ---------------------------------------------------------------------------


class WorkflowExecutor:
    """Execute a v3 workflow."""

    def __init__(
        self,
        workflow_dir_or_workflow: str | Path | Workflow,
        *,
        tool_registry: Any,
        skill_registry: Any | None = None,
        policy_executor: Any | None = None,
        observation_poll_fn: Callable[[], Any] | None = None,
        observation_hz: float = 10.0,
        trace_dir: str | Path | None = None,
        checkpoints: str = "off",
        world_snapshot_fn: Callable[[], Any] | None = None,
        subgraph_exit_hook: Callable[[SubgraphExitEvent], None] | None = None,
        max_node_workers: int = 8,
        node_visit_cap: int | None = None,
    ):
        if isinstance(workflow_dir_or_workflow, Workflow):
            self.workflow: Workflow = workflow_dir_or_workflow
            self.workflow_dir = Path(self.workflow.workflow_dir)
        else:
            path = Path(workflow_dir_or_workflow)
            wf_path = path / "workflow.json" if path.is_dir() else path
            self.workflow = load_workflow(wf_path)
            self.workflow_dir = wf_path.parent

        self.tool_registry = tool_registry
        self.skill_registry = skill_registry
        self.policy_executor = policy_executor
        self.observation_poll_fn = observation_poll_fn
        self.observation_hz = float(observation_hz)
        self.max_node_workers = int(max_node_workers)

        # Subgraph-exit hook: harnesses set this to snapshot the World
        # after each subgraph and evaluate its checkpoints. Default
        # ``None`` short-circuits to zero overhead in production. Visit
        # indices are 0-based per sg_name.
        self.subgraph_exit_hook = subgraph_exit_hook
        self._visit_counts: dict[str, int] = {}

        # Cross-subgraph outputs: {subgraph_name: {output_name: value}}.
        # Most-recent run wins.
        self.cross_subgraph_outputs: dict[str, dict[str, Any]] = {}

        # Facade-supplied initial inputs (``gap.execute(..., inputs=...)``).
        # Addressable at the top level as ``{"$ref": "in.<name>"}`` and used
        # as the base producer for subgraph input binding (upstream subgraph
        # outputs of the same name win).
        self.initial_inputs: dict[str, Any] = {}

        # Execution-time enforcement of ``validate=True`` postcondition
        # checkpoints. ``"off"`` skips entirely; ``"warn"`` logs failures
        # and continues; ``"raise"`` raises VerificationFailed. Requires a
        # ``world_snapshot_fn`` (a connector exposing ground truth) — when
        # absent, enforcement degrades to a one-shot warning.
        if checkpoints not in ("off", "warn", "raise"):
            raise ValueError(
                f"checkpoints must be 'off', 'warn', or 'raise', "
                f"got {checkpoints!r}"
            )
        self.checkpoints = checkpoints
        self.world_snapshot_fn = world_snapshot_fn
        self.checkpoint_results: list[Any] = []
        self._checkpoints_warned = False

        if node_visit_cap is not None:
            self.node_visit_cap = int(node_visit_cap)
        else:
            self.node_visit_cap = int(os.environ.get("GAP_ITERATION_CAP", "10000"))

        if trace_dir is None:
            trace_dir = os.environ.get("GAP_TRACE_DIR", "") or self.workflow_dir
        self.trace = DagTrace(trace_dir)

        self.observation_stream: ObservationStream | None = None
        self.skill_instances: dict[str, Any] = {}

        # Status of the end node the last execute() terminated at
        # ("success" | "failure"), or None if it never reached one.
        self.exit_status: str | None = None

        # First node error of the current execute(), kept so the terminal
        # failure can name a cause instead of only an end-node label.
        self._first_node_error: str | None = None

        self._stream_grace_s = float(
            os.environ.get("GAP_PARALLEL_CANCEL_GRACE_S", "2.0"),
        )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def execute(self) -> None:
        """Drive the top-level workflow from START to an end node."""
        guards.reset_counters()
        self.checkpoint_results = []
        self.exit_status = None
        self._first_node_error = None

        issues = validate_workflow(
            self.workflow,
            agent_registry=None,
            skill_registry=self.skill_registry,
            tool_registry=self.tool_registry,
        )
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        for w in warnings:
            logger.warning("Validation warning: %s", w)
        if errors:
            raise GraphValidationError(errors)

        self._initialize_trace_graph()
        self.trace.copy_workflow(self.workflow_dir)

        # The observation stream starts only when the constructor was given a
        # poll fn (the connector's ``get_observation``) **and the workflow
        # actually reads one**. Almost every connector has a ``get_observation``,
        # so the poll fn alone started a background poller for every graph --
        # including the great majority that never bind the stream.
        #
        # On a single world that is a wasted picture now and then. On a wave it
        # is the whole clock: the poller asks for an observation, the fleet
        # answers it with a render of every world and every camera, and because
        # the fleet steps only when every world is parked, that render lands
        # between every pair of steps. Profiled on a twelve-world surface_prep
        # wave: 78% of the wall (3069 s of 3920 s) was the barrier held with the
        # fleet thread inside `render_worlds`, 119 of 119 samples, two thirds of
        # those in the device-to-host copy -- for a graph whose text contains no
        # reference to `observation_stream` at all. Measured: render 340 ms
        # against a 36 ms step, and a state-only observation costs 0.00 ms.
        #
        # It also flattened the point of batching: per-episode cost was 31/27/31
        # ms at 4/6/12 worlds with the render, and 6.6/4.0/3.0 without it.
        #
        # Workflow meta override beats env var; fall back to the constructor's hz.
        # Visual-servo loops bump this to match the servo update_hz.
        if self.observation_poll_fn is not None and self._reads_observation_stream():
            hz_str = (
                self.workflow.meta.get("observation_stream_hz")
                or os.environ.get("GAP_OBSERVATION_STREAM_HZ", "")
            )
            try:
                hz = float(hz_str)
            except (TypeError, ValueError):
                hz = self.observation_hz
            self.observation_stream = start_observation_stream(
                self.observation_poll_fn, hz=hz,
            )

        try:
            self._run_top_level()
        except Exception:
            # A failure before the first node completes would otherwise leave
            # no dag_trace.json at all -- and a reader with no trace cannot
            # tell "the graph aborted at load" from "the graph ran and did
            # nothing", which is exactly the confusion that cost a
            # self-learning run five iterations. The trace already holds the
            # error event; write it out before the exception leaves.
            try:
                self.trace.flush()
            except Exception:  # noqa: BLE001 - never mask the original failure
                logger.debug("could not flush the trace after a failure", exc_info=True)
            raise
        finally:
            if self.observation_stream is not None:
                self.observation_stream.stop()
                self.observation_stream = None
            self.skill_instances.clear()

    def close(self) -> None:
        """Release executor resources. Kept for API parity; the gRPC
        ServiceRegistry this used to close is gone — ``execute()``'s
        ``finally`` already stops the observation stream."""
        if self.observation_stream is not None:
            self.observation_stream.stop()
            self.observation_stream = None

    # ------------------------------------------------------------------
    # Top-level scheduler
    # ------------------------------------------------------------------

    def _run_top_level(self) -> None:
        """Walk the top-level workflow."""
        scope = _ScopeState()
        scope.local_outputs[RESERVED_INPUT_PSEUDOSTATE] = {
            OBSERVATION_STREAM_INPUT_NAME: self.observation_stream,
            **self.initial_inputs,
        }

        terminal = self._run_scope(
            scope=scope,
            nodes=self.workflow.nodes,
            edges=self.workflow.edges,
            cond_edges=self.workflow.conditional_edges,
            scope_name="<workflow>",
        )

        # The terminal node should be type=end.
        if terminal is None:
            raise PipelineError(
                "workflow exited without reaching an end node"
            )
        node = self.workflow.nodes[terminal]
        if node.type != "end":
            raise PipelineError(
                f"workflow terminal node {terminal!r} is not type=end "
                f"(type={node.type})"
            )
        self.exit_status = node.status
        self._run_recovery(terminal, node)
        self.trace.flush()
        logger.info(
            "DAG trace written to %s",
            getattr(self.trace, "_output_dir", "<trace dir>"),
        )
        if node.status == "success":
            logger.info("Workflow terminated at end node %r (success)", terminal)
            return
        detail = (
            f"; first node error -- {self._first_node_error}"
            if self._first_node_error else ""
        )
        raise PipelineError(
            f"workflow terminated at end node {terminal!r} (failure){detail}"
        )

    # ------------------------------------------------------------------
    # Scope scheduler
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_reachable(
        start: str,
        outgoing: dict[str, list[str]],
        cond_edges: dict[str, Any],
    ) -> set[str]:
        """Return the set of nodes forward-reachable from ``start`` (inclusive).

        Follows both static edges (``outgoing``) and conditional-edge mapping
        targets; ``START``/``END`` are excluded. Used to compute a loop body
        when a backward (conditional) edge re-enters a completed node.
        """
        seen: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen or cur in (START, END):
                continue
            seen.add(cur)
            for dst in outgoing.get(cur, []):
                if dst not in seen:
                    stack.append(dst)
            ce = cond_edges.get(cur)
            if ce is not None:
                for dst in ce.mapping.values():
                    if dst not in seen:
                        stack.append(dst)
        seen.discard(START)
        seen.discard(END)
        return seen

    def _run_scope(
        self,
        scope: _ScopeState,
        nodes: dict[str, NodeDef],
        edges: tuple[tuple[str, str], ...],
        cond_edges: dict[str, Any],
        scope_name: str,
    ) -> str | None:
        """Run a single scope (workflow or subgraph) to completion.

        Returns the name of the terminal node reached (one whose outgoing
        edge is to ``END``), or ``None`` if the scope ended via an
        ``END`` target on a streaming-only path. Streaming nodes are
        spawned eagerly and torn down on scope exit.
        """
        # Outgoing-edge index for fast frontier advancement.
        outgoing: dict[str, list[str]] = {}
        for src, dst in edges:
            outgoing.setdefault(src, []).append(dst)

        # Initial frontier: targets of START edges (deduplicated).
        frontier: list[str] = []
        seen_in_frontier: set[str] = set()
        for src, dst in edges:
            if src == START and dst not in seen_in_frontier:
                frontier.append(dst)
                seen_in_frontier.add(dst)

        terminal_node: str | None = None
        super_step = 0

        try:
            while frontier:
                super_step += 1
                if super_step > self.node_visit_cap:
                    raise PipelineError(
                        f"scope {scope_name!r} exceeded super-step cap "
                        f"{self.node_visit_cap}; possible runaway loop"
                    )

                # Split frontier into streaming / regular nodes. Streaming
                # nodes are spawned and immediately removed from frontier.
                ready_regular: list[str] = []
                next_frontier: list[str] = []
                end_reached = False
                for node_name in frontier:
                    if node_name == END:
                        # Reaching END from a regular edge ends the scope.
                        # The terminal_node was set by whichever predecessor
                        # had an outgoing edge to END (see fold loop below).
                        end_reached = True
                        continue
                    if node_name not in nodes:
                        raise PipelineError(
                            f"scope {scope_name!r}: edge target {node_name!r} "
                            f"not declared"
                        )
                    node = nodes[node_name]
                    if node.streaming:
                        self._spawn_streaming(node_name, node, scope)
                        continue
                    if node_name in scope.completed_nodes:
                        # Already ran; do not re-execute. Frontier dedup
                        # plus completed_nodes guard means re-entry is a
                        # cycle attempt — silently skip.
                        continue
                    ready_regular.append(node_name)

                if end_reached and not ready_regular:
                    return terminal_node

                # Run all ready_regular nodes concurrently.
                results = self._run_super_step(
                    ready_regular, nodes, scope, scope_name,
                )

                # Fold results back; advance frontier.
                for node_name, result in results.items():
                    scope.local_outputs[node_name] = result
                    scope.completed_nodes.add(node_name)
                    node = nodes[node_name]

                    # End nodes terminate the scope (top-level workflow).
                    if node.type == "end":
                        terminal_node = node_name
                        return terminal_node

                    # Static outgoing edges.
                    for dst in outgoing.get(node_name, []):
                        if dst == END:
                            # This node is the subgraph's terminal — its
                            # name (or output field) becomes the exit value.
                            terminal_node = node_name
                            continue
                        if dst not in next_frontier and dst not in seen_in_frontier:
                            next_frontier.append(dst)
                            seen_in_frontier.add(dst)
                        elif dst not in next_frontier:
                            next_frontier.append(dst)

                    # Conditional edges.
                    if node_name in cond_edges:
                        tgt = self._resolve_cond_target(
                            node_name, node, scope, cond_edges[node_name],
                        )
                        if tgt is None:
                            continue
                        if tgt == END:
                            terminal_node = node_name
                            continue
                        if tgt in scope.completed_nodes:
                            # Backward (loop) edge: a conditional edge resolved
                            # to a node that already ran in this scope. Reset
                            # the loop body — every node forward-reachable from
                            # the target — so it re-executes, then re-enqueue
                            # the target. (Static edges to completed nodes keep
                            # the silent-skip dedup; only conditional edges form
                            # loops.) local_outputs are left intact: re-runs
                            # overwrite them, matching the documented
                            # "most recent run wins" semantics. The super-step
                            # cap still bounds runaway loops.
                            body = self._forward_reachable(
                                tgt, outgoing, cond_edges,
                            )
                            for n in body:
                                scope.completed_nodes.discard(n)
                                seen_in_frontier.discard(n)
                            if tgt not in next_frontier:
                                next_frontier.append(tgt)
                            seen_in_frontier.add(tgt)
                            continue
                        if tgt not in next_frontier:
                            next_frontier.append(tgt)

                # If END was the only thing in the prior frontier and we
                # ran no regular nodes this step, exit now.
                if end_reached and not next_frontier:
                    return terminal_node

                frontier = next_frontier

            return terminal_node
        finally:
            self._teardown_streaming(scope)

    # ------------------------------------------------------------------
    # Super-step batch
    # ------------------------------------------------------------------

    def _run_super_step(
        self,
        ready: list[str],
        nodes: dict[str, NodeDef],
        scope: _ScopeState,
        scope_name: str,
    ) -> dict[str, Any]:
        """Run one super-step batch of regular (non-streaming) nodes.

        All nodes in ``ready`` execute concurrently. The first to raise
        cancels the others (they have no inter-node dependencies within
        a super-step).
        """
        if not ready:
            return {}

        if len(ready) == 1:
            # Hot path: avoid threading overhead for sequential graphs.
            name = ready[0]
            return {name: self._dispatch_node(name, nodes[name], scope, scope_name)}

        results: dict[str, Any] = {}
        first_error: Exception | None = None
        with ThreadPoolExecutor(max_workers=self.max_node_workers) as pool:
            futures: dict[Future, str] = {}
            for name in ready:
                fut = pool.submit(
                    self._dispatch_node, name, nodes[name], scope, scope_name,
                )
                futures[fut] = name
            for fut in list(futures.keys()):
                node_name = futures[fut]
                try:
                    results[node_name] = fut.result()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                    for other in futures:
                        if other is not fut and not other.done():
                            other.cancel()
        if first_error is not None:
            raise first_error
        return results

    # ------------------------------------------------------------------
    # Per-node dispatch
    # ------------------------------------------------------------------

    def _dispatch_node(
        self,
        node_name: str,
        node: NodeDef,
        scope: _ScopeState,
        scope_name: str,
    ) -> Any:
        """Resolve inputs and run a single (non-streaming) node body."""
        full_id = f"{scope_name}.{node_name}" if scope_name != "<workflow>" else node_name

        # Subgraph nodes recurse.
        if node.type == "subgraph":
            return self._run_subgraph(node_name, node, scope)

        # Router nodes handle Send dispatch.
        if node.type == "router":
            return self._run_router(full_id, node_name, node, scope)

        # End nodes are terminal markers; their "result" is the recovery list.
        if node.type == "end":
            return {"_end": True, "status": node.status}

        # Noop: passthrough marker, used as a named subgraph terminal.
        if node.type == "noop":
            return {}

        self.trace.start_node(full_id)
        success = False
        error_text: str | None = None
        try:
            resolved = resolve_inputs(node.inputs, scope.local_outputs)
            self.trace.record_resolved_inputs(full_id, resolved)

            result: Any
            if node.type == "tool":
                result = execute_tool_node(
                    full_id, node, resolved, self.tool_registry,
                    policy_executor=self.policy_executor,
                )
                self.trace.record_output(full_id, result)
            elif node.type == "script":
                bundle_name = self._lookup_bundle_for_scope(scope_name)
                result = execute_script_node(
                    full_id, node, resolved, self.tool_registry, self.workflow_dir,
                    trace=self.trace,
                    policy_executor=self.policy_executor,
                    bundle_name=bundle_name,
                    skill_registry=self.skill_registry,
                )
                self.trace.record_output(full_id, result)
            else:
                raise PipelineError(
                    f"node {full_id!r} has unknown type {node.type!r}"
                )
            success = True
            return result
        except Exception as exc:
            # The message is the only record of *why* a node failed, and a
            # failure is routed, not propagated: `_run_subgraph` catches this
            # and takes the `on_error` exit. Without recording it here the
            # reason survives only in the log stream, so every consumer of
            # dag_trace.json -- the visualizer, and any feedback loop reading
            # the trace -- sees `status: error, error_message: null` and has
            # to guess. Record, then let it propagate unchanged.
            error_text = _node_error_text(exc)
            raise
        finally:
            self.trace.end_node(full_id, success)
            if error_text is not None:
                self.trace.record_error(full_id, error_text)
                if self._first_node_error is None:
                    self._first_node_error = f"{full_id}: {error_text}"
            # Incremental flush: a crash mid-run (segfault, OOM kill, disk
            # full during video render) previously lost the WHOLE trace —
            # node_data/ was on disk but dag_trace.json never existed, so
            # the run was invisible to the visualizer. Never let the flush
            # itself break execution.
            try:
                self.trace.flush()
            except Exception:
                logger.debug("incremental trace flush failed", exc_info=True)

    def _lookup_bundle_for_scope(self, scope_name: str) -> str:
        """Map a scope name back to the owning subgraph's `skill` field."""
        if scope_name == "<workflow>":
            return ""
        sg = self.workflow.subgraphs.get(scope_name)
        if sg is None:
            return ""
        return sg.skill or ""

    # ------------------------------------------------------------------
    # Subgraph recursion
    # ------------------------------------------------------------------

    def _run_subgraph(
        self, node_name: str, node: NodeDef, parent_scope: _ScopeState,
    ) -> dict[str, Any]:
        """Recurse into a subgraph; bind inputs, run inner scope, bind outputs."""
        sg_name = node.ref or ""
        sg = self.workflow.subgraphs.get(sg_name)
        if sg is None:
            raise PipelineError(
                f"subgraph node {node_name!r} references unknown subgraph "
                f"{sg_name!r}"
            )

        logger.info("--- Subgraph '%s' ---", sg_name)
        try:
            bound_inputs = self._bind_subgraph_inputs(
                sg_name, sg, node, parent_scope,
            )
        except PipelineError as exc:
            # Blame the node, not the workflow: a binding failure is the one
            # error that happens before any node runs, so without this the
            # trace names nothing and the failure reads as "the graph ran and
            # did nothing".
            self.trace.record_error(node_name, str(exc))
            raise

        scope = _ScopeState()
        scope.local_outputs[RESERVED_INPUT_PSEUDOSTATE] = bound_inputs

        visit_start = time.perf_counter()
        terminal: str | None
        error_path = False
        try:
            terminal = self._run_scope(
                scope=scope,
                nodes=sg.nodes,
                edges=sg.edges,
                cond_edges=sg.conditional_edges,
                scope_name=sg_name,
            )
        except Exception as exc:
            if sg.on_error is None or is_terminal(exc):
                # Terminal means the world this graph acts on is gone -- the
                # episode ended under it. Routing would send the graph around
                # its recovery edge to call another tool that raises the same
                # thing, forever. See `gap_core.errors.is_terminal`.
                raise
            logger.warning(
                "subgraph %r raised %s; routing to on_error=%r",
                sg_name, exc, sg.on_error,
            )
            terminal = None
            error_path = True

        # Compute the subgraph's exit condition.
        exit_value: str
        if error_path:
            exit_value = sg.on_error or ""
        elif sg.exit.router_field is None:
            exit_value = terminal or ""
        else:
            if terminal is None or terminal not in scope.local_outputs:
                raise PipelineError(
                    f"subgraph {sg_name!r}: cannot resolve exit "
                    f"(no terminal node output)"
                )
            term_out = scope.local_outputs[terminal]
            exit_value = self._read_field(term_out, sg.exit.router_field)
            if not isinstance(exit_value, str):
                exit_value = str(exit_value)

        legal_exits = set(sg.exit.success_values)
        if sg.on_error is not None:
            legal_exits.add(sg.on_error)
        if legal_exits and exit_value not in legal_exits:
            raise PipelineError(
                f"subgraph {sg_name!r}: exit value {exit_value!r} not in "
                f"declared success_values {list(sg.exit.success_values)!r} "
                f"and is not on_error={sg.on_error!r}"
            )

        # Bind subgraph outputs.
        bound_outputs: dict[str, Any] = {}
        for out_name, ref in sg.outputs.items():
            try:
                bound_outputs[out_name] = resolve_ref(ref, scope.local_outputs)
            except Exception as exc:
                logger.debug(
                    "Subgraph %r output %r unbound on this exit (%s); "
                    "downstream consumers will see no value",
                    sg_name, out_name, exc,
                )
        self.cross_subgraph_outputs[sg_name] = bound_outputs

        # Execution-time enforcement of this subgraph's ``validate=True``
        # postcondition checkpoints. Run only on a normal exit; on the
        # ``on_error`` path the postcondition is moot.
        if self.checkpoints != "off" and not error_path:
            self._run_validate_checkpoints(sg_name, sg, scope, bound_outputs)

        # Fire the exit hook before returning. `_run_scope`'s finally
        # already drained streaming futures via `_teardown_streaming`, so
        # the world is quiescent when the harness snapshots it. Hook
        # errors propagate — the rehearsal harness owns its own
        # exception boundary.
        if self.subgraph_exit_hook is not None:
            visit_index = self._visit_counts.get(sg_name, 0)
            self._visit_counts[sg_name] = visit_index + 1
            try:
                self.subgraph_exit_hook(SubgraphExitEvent(
                    sg_name=sg_name,
                    visit_index=visit_index,
                    bound_outputs=dict(bound_outputs),
                    exit_value=exit_value,
                    error_path=error_path,
                    elapsed_s=time.perf_counter() - visit_start,
                ))
            except Exception:
                logger.exception(
                    "subgraph_exit_hook raised for sg=%r visit=%d; "
                    "execution continues",
                    sg_name, visit_index,
                )

        # The subgraph node's "result" is the bound outputs plus the exit
        # condition under a reserved key. Top-level conditional_edges on
        # the subgraph node read `_exit` to dispatch.
        return {**bound_outputs, "_exit": exit_value}

    def _run_validate_checkpoints(
        self,
        sg_name: str,
        sg: SubgraphDef,
        scope: _ScopeState,
        bound_outputs: dict[str, Any],
    ) -> None:
        """Enforce ``validate=True`` postcondition checkpoints at execution time.

        Loads the subgraph's checkpoint sidecar
        ``<workflow_dir>/checkpoints/<sg_name>.py`` (when present) via
        :func:`gap.runtime.verify.load_checkpoints`, takes a ground-truth
        :class:`gap.runtime.verify.World` snapshot via
        ``world_snapshot_fn`` (the connector's ``world_snapshot``), and
        evaluates every ``validate=True`` predicate against
        ``(world, bound_outputs)``. ``validate=False`` probes are never
        enforced here.

        Results accumulate on ``self.checkpoint_results``. Failures are
        logged when ``checkpoints == "warn"`` and raise
        :class:`gap.errors.VerificationFailed` (naming the failing
        checkpoints) when ``checkpoints == "raise"``.

        When no ``world_snapshot_fn`` is available (connector exposes no
        ground truth — e.g. a real robot), enforcement degrades to a
        one-shot warning and the run proceeds unchanged.
        """
        if self.world_snapshot_fn is None:
            if not self._checkpoints_warned:
                self._checkpoints_warned = True
                logger.warning(
                    "checkpoints=%r but no world_snapshot_fn is available "
                    "(connector exposes no ground truth); checkpoint "
                    "enforcement is skipped",
                    self.checkpoints,
                )
            return

        cp_path = self.workflow_dir / "checkpoints" / f"{sg_name}.py"
        if not cp_path.exists():
            return

        from gap.runtime.verify import evaluate_checkpoint, load_checkpoints

        checkpoints = load_checkpoints(cp_path)
        world = self.world_snapshot_fn()
        failed: list[str] = []
        for cp in checkpoints:
            if not cp.validate:
                continue
            result = evaluate_checkpoint(cp, world, bound_outputs)
            self.checkpoint_results.append(result)
            if not result.passed:
                failed.append(cp.name)

        if not failed:
            return
        if self.checkpoints == "raise":
            raise VerificationFailed(
                f"subgraph {sg_name!r}: {len(failed)} validate=True "
                f"checkpoint(s) failed: {failed}"
            )
        logger.warning(
            "subgraph %r: %d validate=True checkpoint(s) failed: %s "
            "(checkpoints='warn'; continuing)",
            sg_name, len(failed), failed,
        )

    def _bind_subgraph_inputs(
        self,
        sg_name: str,
        sg: SubgraphDef,
        node: NodeDef | None = None,
        scope: _ScopeState | None = None,
    ) -> dict[str, Any]:
        """Bind a subgraph's declared inputs at one call site.

        Three sources, in precedence order, because a subgraph is a callee
        and its caller is entitled to say what it is called with:

        1. **The calling node's own ``inputs``** -- a literal, or a ``$ref``
           into the caller's scope. WORKFLOW_FORMAT tells authors "literals
           are passed as plain JSON", and script nodes have always honoured
           them; a subgraph node ignoring them was the difference between a
           graph the validator accepts and a graph that raises before its
           first node. Measured: a workflow whose entry subgraph declared
           ``arm_id`` and was called with ``{"inputs": {"arm_id": 0}}``
           aborted in 60 of 60 episodes across five self-learning
           iterations, and every episode was reported to the improver as a
           near miss.
        2. The latest cross-subgraph output of the same name.
        3. The facade's ``initial_inputs``.

        A name none of the three answers is still an error: a subgraph
        cannot run without a declared input, and failing here names the
        input, where failing later names whatever the script did with
        ``None``.

        Each call-site input is resolved **on its own**, and one that cannot
        be resolved falls through to the next source rather than failing the
        call. Before these inputs were honoured at all they were ignored
        wholesale, so a graph carrying a stale ``$ref`` beside inputs its
        upstream subgraphs already produce ran perfectly well; resolving the
        dict as a unit turned every such graph into a load-time abort. The
        shipped expert tool-hanging graph is one: its realign node passes
        ``$ref realign.held_object_in_tcp``, no node produces that field, and
        the six of eight episodes it used to hang became zero.
        """
        site: dict[str, Any] = {}
        unresolved: dict[str, str] = {}
        if node is not None and node.inputs:
            local = scope.local_outputs if scope is not None else {}
            for in_name, spec in node.inputs.items():
                if in_name not in sg.inputs:
                    continue
                try:
                    site[in_name] = resolve_inputs({in_name: spec}, local)[in_name]
                except Exception as exc:  # noqa: BLE001 - recorded, then the next source is tried
                    unresolved[in_name] = str(exc)
        bound: dict[str, Any] = {}
        for in_name in sg.inputs:
            value = site.get(in_name)
            if value is None:
                value = self._lookup_cross_subgraph_output(in_name)
            if value is None:
                detail = (
                    f"; the calling node passes one but it does not resolve ({unresolved[in_name]})"
                    if in_name in unresolved
                    else " and the calling node passes no value for it"
                )
                raise PipelineError(
                    f"subgraph {sg_name!r} input {in_name!r} has no upstream producer{detail}"
                )
            bound[in_name] = value
        return bound

    def _reads_observation_stream(self) -> bool:
        """Whether any node in this workflow binds the observation stream.

        A node reads it by naming :data:`OBSERVATION_STREAM_INPUT_NAME` among
        its inputs, or a subgraph by declaring it as one of its own -- both are
        checked, because a subgraph's binding is what its nodes then resolve
        against. Streaming nodes are included: a ``streaming`` node is exactly
        the kind that wants a live feed.

        Conservative by construction: anything that mentions the name at all
        keeps the poller. The cost of a false positive is the status quo; the
        cost of a false negative is a graph that silently reads a stream nobody
        is filling, so the check errs toward starting it.
        """
        name = OBSERVATION_STREAM_INPUT_NAME
        if name in (self.initial_inputs or {}):
            return True
        for node in self.workflow.nodes.values():
            if name in (getattr(node, "inputs", None) or {}):
                return True
            if getattr(node, "streaming", False):
                return True
        for sg in self.workflow.subgraphs.values():
            if name in (getattr(sg, "inputs", None) or {}):
                return True
            for node in sg.nodes.values():
                if name in (getattr(node, "inputs", None) or {}):
                    return True
                if getattr(node, "streaming", False):
                    return True
        return False

    def _lookup_cross_subgraph_output(self, name: str) -> Any | None:
        if name == OBSERVATION_STREAM_INPUT_NAME and self.observation_stream is not None:
            return self.observation_stream
        latest: Any | None = self.initial_inputs.get(name)
        for sg_outputs in self.cross_subgraph_outputs.values():
            if name in sg_outputs:
                latest = sg_outputs[name]
        return latest

    # ------------------------------------------------------------------
    # Conditional edges
    # ------------------------------------------------------------------

    def _resolve_cond_target(
        self,
        node_name: str,
        node: NodeDef,
        scope: _ScopeState,
        ce: Any,
    ) -> str | None:
        """Resolve a conditional edge to a target name."""
        if node.type == "router":
            # Router nodes return their dispatch result through their output.
            # The router_field is None; the router's output IS the target name
            # (or, for Send, a list — handled by _run_router which folds the
            # list into local_outputs and never reaches conditional_edges).
            tgt = scope.local_outputs.get(node_name)
            if isinstance(tgt, str) and tgt in ce.mapping:
                return ce.mapping[tgt]
            raise PipelineError(
                f"router {node_name!r} produced {tgt!r}, not a valid mapping key"
            )

        # Non-router source: read router_field on the node's output.
        out = scope.local_outputs.get(node_name)
        if out is None:
            raise PipelineError(
                f"conditional_edges source {node_name!r} has no output"
            )
        # For subgraph nodes, _exit is the canonical router field.
        rf = ce.router_field
        if node.type == "subgraph" and rf == "exit":
            rf = "_exit"
        value = self._read_field(out, rf or "")
        if not isinstance(value, str):
            value = str(value)
        if value not in ce.mapping:
            raise PipelineError(
                f"conditional_edges from {node_name!r}: value {value!r} "
                f"not in mapping {list(ce.mapping.keys())!r}"
            )
        return ce.mapping[value]

    @staticmethod
    def _read_field(out: Any, field_name: str) -> Any:
        if isinstance(out, dict):
            if field_name in out:
                return out[field_name]
        if hasattr(out, field_name):
            return getattr(out, field_name)
        raise PipelineError(
            f"cannot read field {field_name!r} from output of type "
            f"{type(out).__name__}"
        )

    # ------------------------------------------------------------------
    # Streaming nodes
    # ------------------------------------------------------------------

    def _spawn_streaming(
        self, node_name: str, node: NodeDef, scope: _ScopeState,
    ) -> None:
        """Spawn a streaming node into a detached future."""
        if node_name in scope.streaming_slots:
            return  # idempotent
        cancel_token = CancelToken()
        slot = StreamSlot(node_name, cancel_token)

        # Resolve inputs eagerly. The skill itself is responsible for
        # iterating and publishing to the slot via ctx (see below).
        resolved = resolve_inputs(node.inputs, scope.local_outputs)

        # Place the slot in local_outputs BEFORE spawning so that
        # downstream nodes resolving $ref to this name during the same
        # super-step see the slot (latest() will block until first
        # publish).
        scope.local_outputs[node_name] = slot
        scope.streaming_slots[node_name] = slot

        full_id = node_name
        pool = ThreadPoolExecutor(max_workers=1)

        def _run() -> Any:
            # The convention for streaming skills: the skill loop calls
            # ctx.publish(...) on each iteration. The node executors bind
            # the active StreamSlot onto the NodeContext they construct so
            # the skill code stays ignorant of the slot itself.
            try:
                if node.type == "tool":
                    return execute_tool_node(
                        full_id, node, resolved, self.tool_registry,
                        policy_executor=self.policy_executor,
                        cancel_token=cancel_token,
                        stream_slot=slot,
                    )
                if node.type == "script":
                    return execute_script_node(
                        full_id, node, resolved, self.tool_registry, self.workflow_dir,
                        trace=self.trace,
                        policy_executor=self.policy_executor,
                        cancel_token=cancel_token,
                        bundle_name=self._lookup_bundle_for_scope("<workflow>"),
                        skill_registry=self.skill_registry,
                        stream_slot=slot,
                    )
                raise PipelineError(
                    f"streaming nodes must be type=tool or type=script, "
                    f"got {node.type!r}"
                )
            except TaskCancelled:
                logger.debug("streaming node %r cancelled", node_name)
                return None

        fut = pool.submit(_run)
        scope.streaming_futures[node_name] = fut
        # Detach the pool — the future keeps it alive until completion;
        # we'll cancel via the token on teardown.
        pool.shutdown(wait=False)

    def _teardown_streaming(self, scope: _ScopeState) -> None:
        """Cancel and drain all streaming nodes in a scope."""
        if not scope.streaming_slots:
            return
        for slot in scope.streaming_slots.values():
            slot.cancel()
        # Wait briefly for cooperative shutdown.
        wait(
            list(scope.streaming_futures.values()),
            timeout=self._stream_grace_s,
        )
        for name, fut in scope.streaming_futures.items():
            if not fut.done():
                fut.cancel()
                logger.warning("streaming node %r force-cancelled", name)

    # ------------------------------------------------------------------
    # Router / Send dispatch
    # ------------------------------------------------------------------

    def _run_router(
        self, full_id: str, node_name: str, node: NodeDef, scope: _ScopeState,
    ) -> Any:
        """Run a router node. Returns either a string target or a list of Send results."""
        resolved = resolve_inputs(node.inputs, scope.local_outputs)

        # Run the routing script. Return value: str OR list[dict{to,inputs}].
        result = execute_script_node(
            full_id, node, resolved, self.tool_registry, self.workflow_dir,
            trace=self.trace, policy_executor=self.policy_executor,
            bundle_name="", skill_registry=self.skill_registry,
        )
        # execute_script_node returns a dict; routers should put their
        # decision under a `route` key.
        if not isinstance(result, dict) or "route" not in result:
            raise PipelineError(
                f"router {full_id!r} script must return a dict with 'route' "
                f"key; got {type(result).__name__}"
            )
        route = result["route"]

        # Static dispatch: route is a string target name.
        if isinstance(route, str):
            return route

        # Send dispatch: route is a list of {to, inputs} dicts.
        if isinstance(route, list):
            collected: list[Any] = []
            with ThreadPoolExecutor(max_workers=self.max_node_workers) as pool:
                futures: list[Future] = []
                for i, send in enumerate(route):
                    if not isinstance(send, dict) or "to" not in send:
                        raise PipelineError(
                            f"router {full_id!r}: route[{i}] must be "
                            f"{{'to', 'inputs'}}"
                        )
                    target_name = send["to"]
                    target_inputs = send.get("inputs", {}) or {}
                    # The target node is found in the same scope; spawn one
                    # copy with scoped inputs.
                    futures.append(pool.submit(
                        self._dispatch_send_copy,
                        full_id, i, target_name, target_inputs, scope,
                    ))
                for fut in futures:
                    collected.append(fut.result())
            return collected

        raise PipelineError(
            f"router {full_id!r}: 'route' must be string or list, got "
            f"{type(route).__name__}"
        )

    def _dispatch_send_copy(
        self,
        router_full_id: str,
        idx: int,
        target_name: str,
        target_inputs: dict[str, Any],
        scope: _ScopeState,
    ) -> Any:
        """Dispatch one Send copy of a target node with scoped inputs."""
        # The target must be a declared node in the scope's nodes dict; we
        # find it by walking the workflow. (Routers operate at the scope
        # level, so we need a handle on the scope's nodes — pass via scope
        # in a future refactor; for now, scan workflow + subgraphs.)
        node = self.workflow.nodes.get(target_name)
        if node is None:
            for sg in self.workflow.subgraphs.values():
                node = sg.nodes.get(target_name)
                if node is not None:
                    break
        if node is None:
            raise PipelineError(
                f"Send from {router_full_id!r}[{idx}]: unknown target "
                f"{target_name!r}"
            )

        # Build a one-off scope view: parent local_outputs + target-specific
        # inputs under a synthetic key. The simplest path: synthesize a
        # NodeDef whose inputs override the target's declared inputs.
        sub_node = NodeDef(
            type=node.type,
            inputs={**node.inputs, **target_inputs},
            streaming=node.streaming,
            script=node.script, tool=node.tool,
            ref=node.ref,
        )
        return self._dispatch_node(
            f"{target_name}#{idx}", sub_node, scope, "<send>",
        )

    # ------------------------------------------------------------------
    # End nodes
    # ------------------------------------------------------------------

    def _run_recovery(self, node_name: str, node: NodeDef) -> None:
        """Run an end node's best-effort recovery tool calls. Never raises."""
        if not node.recovery:
            return
        logger.info(
            "End node %r (%s): running %d recovery action(s)",
            node_name, node.status, len(node.recovery),
        )
        ctx = NodeContext(
            self.tool_registry, trace=self.trace,
            node_id=f"{node_name}.recovery",
            policy_executor=self.policy_executor,
        )
        for i, action in enumerate(node.recovery):
            try:
                ctx.tool(action.tool, **dict(action.inputs))
                logger.info("  recovery[%d] %s OK", i, action.tool)
            except Exception as exc:
                logger.warning(
                    "  recovery[%d] %s failed: %s", i, action.tool, exc,
                )

    # ------------------------------------------------------------------
    # Trace graph initialization
    # ------------------------------------------------------------------

    def _initialize_trace_graph(self) -> None:
        """Register workflow nodes and edges in the trace."""
        for node_name, node in self.workflow.nodes.items():
            self.trace.add_node(node_name, {
                "type": node.type,
                "ref": node.ref,
                "status": node.status,
            })
        for src, dst in self.workflow.edges:
            if src != START and dst != END:
                self.trace.add_edge(src, dst)
        for src, ce in self.workflow.conditional_edges.items():
            for tgt in ce.mapping.values():
                if tgt != END:
                    self.trace.add_edge(src, tgt)

        for sg_name, sg in self.workflow.subgraphs.items():
            for node_name, node in sg.nodes.items():
                full_id = f"{sg_name}.{node_name}"
                self.trace.add_node(full_id, {
                    "type": node.type,
                    "tool": node.tool,
                    "script": node.script,
                    "streaming": node.streaming,
                })
            for src, dst in sg.edges:
                if src != START and dst != END:
                    self.trace.add_edge(
                        f"{sg_name}.{src}", f"{sg_name}.{dst}",
                    )
            for src, ce in sg.conditional_edges.items():
                for tgt in ce.mapping.values():
                    if tgt != END:
                        self.trace.add_edge(
                            f"{sg_name}.{src}", f"{sg_name}.{tgt}",
                        )
