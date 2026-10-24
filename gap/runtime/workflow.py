"""v3 workflow schema: LangGraph-style nodes/edges/conditional-edges.

A workflow is a graph of nodes connected by edges. Both the top-level
workflow and each subgraph have the same shape:

    {
      "version": 3,
      "meta": {...},
      "nodes": { name: NodeDef, ... },
      "edges": [ [src, dst], ... ],
      "conditional_edges": { src: { router_field, mapping }, ... },
      "subgraphs": { name: SubgraphDef, ... }   # only at top level
    }

`START` and `END` are virtual node names. Multiple outgoing edges from one
node = parallel super-step. Fan-in is per-input via `{"$ref": "..."}` —
no reducers, no channels.

A subgraph additionally declares typed `inputs` (cross-subgraph data
binding via the reserved `in.<name>` pseudostate) and `outputs` bound to
internal node fields, plus an `exit` declaration that names the field
on the subgraph's terminal-node output that becomes the subgraph's exit
condition for the top-level conditional edge.

Streaming nodes carry `streaming: true`. They are spawned into a
detached future, never block downstream readiness, and consumers reach
them via `{"$ref": "<node>"}` to get a snapshot of the latest published
value. They have no outgoing edges. The skill they invoke must have
`contract.streaming: true` in its bundle metadata; the validator
cross-checks.

Send (dynamic fan-out) is expressed as a `router` node whose body
returns either a string (target name) or a list of `{to, inputs}` dicts.
Each Send spawns one copy of the target with scoped inputs; outputs
collect into a list under the router node's name.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from gap_core.errors import WorkflowValidationError

logger = logging.getLogger(__name__)

SUPPORTED_VERSION = 3

# Virtual node names.
START = "START"
END = "END"

# Reserved pseudo-state name for subgraph-bound inputs:
#   {"$ref": "in.<input_name>"}
RESERVED_INPUT_PSEUDOSTATE = "in"

# Reserved input name for the executor-injected observation stream.
OBSERVATION_STREAM_INPUT_NAME = "observation_stream"


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ref:
    """Tagged-object reference. JSON form: {"$ref": "head.field.subfield"}.

    Resolved at runtime against a per-subgraph local-outputs dict via
    ``resolve_ref``. The head is either a node name in the current
    subgraph or the reserved pseudostate `in` (for bound subgraph inputs).
    """

    path: str

    def parts(self) -> list[str]:
        """Path segments. Python-style indexing is normalized to dotted
        segments (``arms[0].ee_pose`` == ``arms.0.ee_pose``) — generated
        graphs emit both spellings."""
        normalized = self.path.replace("[", ".").replace("]", "")
        return [p for p in normalized.split(".") if p != ""]


def is_ref_dict(value: Any) -> bool:
    """True if ``value`` is a ``{"$ref": "..."}`` JSON object."""
    return (
        isinstance(value, dict)
        and len(value) == 1
        and "$ref" in value
        and isinstance(value["$ref"], str)
    )


# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------


NodeType = Literal[
    "tool", "script", "router", "subgraph", "end", "noop",
]


@dataclass(frozen=True)
class ToolCall:
    """One recovery action attached to an end node.

    Recovery actions live outside the node-dispatch system — they are a
    short list of best-effort tool calls run when the workflow lands on a
    failure end node (e.g. open the gripper, home the robot). ``tool``
    names the flat dispatch name directly — the same namespace as the
    ``tool`` field on `type: tool` nodes.
    """

    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeDef:
    """A single graph node.

    The ``tool`` field carries the dispatch name for ``type: tool`` nodes
    (the unified dispatch surface — connector methods, atomic skills,
    learned policies, and Python tools are all named here in flat form).
    The ``script`` field carries the path for ``type: script`` and
    ``type: router`` nodes. The ``ref`` field carries the subgraph
    reference for ``type: subgraph`` nodes. Other fields default to None.

    The ``streaming`` flag is only valid on ``tool`` and ``script`` nodes;
    the validator enforces consistency with the chosen tool's contract.

    For ``end`` nodes (top-level workflow termination), ``status`` is
    ``"success"`` or ``"failure"`` and ``recovery`` is a tuple of
    best-effort tool calls to run before the workflow exits.

    For ``router`` nodes (Send dispatch), ``script`` names the routing
    function; the function returns either a string target or a list of
    ``{"to": <node>, "inputs": {...}}`` dicts.

    For ``subgraph`` nodes, ``ref`` names an entry in the workflow's
    top-level ``subgraphs`` dict.
    """

    type: NodeType
    inputs: dict[str, Any] = field(default_factory=dict)   # values: literal | Ref
    streaming: bool = False                                # only on tool/script
    # type-specific dispatch fields:
    script: str | None = None
    tool: str | None = None
    ref: str | None = None                                 # subgraph node target
    # end-node fields:
    status: Literal["success", "failure"] | None = None
    recovery: tuple[ToolCall, ...] = ()


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionalEdge:
    """Conditional dispatch from a single source node.

    The router function on the source node (or, for non-router nodes, a
    field on the source's output) selects one of the mapping's targets.
    For non-router sources, ``router_field`` names the output field whose
    value matches the mapping keys.
    """

    router_field: str | None
    mapping: dict[str, str]

    def targets(self) -> list[str]:
        return list(self.mapping.values())


@dataclass(frozen=True)
class ExitDecl:
    """Subgraph exit declaration.

    When a subgraph's terminal node (the one whose edge points to ``END``)
    completes, ``router_field`` names the field on its output that
    becomes the subgraph's exit condition; the top-level workflow's
    conditional edge on the subgraph node uses this to pick the next
    subgraph.

    ``success_values`` enumerates the legal *success-path* exit
    conditions. When ``router_field is None``, each success value MUST
    be a declared ``noop`` node in the subgraph (the terminal node's
    name becomes the exit). When ``router_field`` is a string, each
    success value is a string the terminal node returns under that
    field — and MUST NOT be a node name.

    The single failure exit lives in ``SubgraphDef.on_error`` (a sibling
    of ``exit:``). It is never a node and never a ``conditional_edges``
    target.
    """

    router_field: str | None
    success_values: tuple[str, ...]

    @property
    def values(self) -> tuple[str, ...]:
        """Backward-compat alias used by the executor's membership check.

        Returns just the success values; the executor adds ``on_error``
        separately when validating a resolved exit_value.
        """
        return self.success_values


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgraphDef:
    """A self-contained inner graph.

    Has the same nodes/edges/conditional-edges shape as the top-level
    workflow plus declared cross-subgraph inputs/outputs and an exit
    declaration.

    ``on_error`` (optional) names an exit value emitted when any node in
    the subgraph raises. Without this field, exceptions propagate and
    crash the workflow. With it, the exception is caught, the
    declared value is used as the subgraph's exit condition (bypassing
    the normal terminal-node + router_field path), and the top-level
    conditional edge routes accordingly. Use this for "any failure →
    failure exit" subgraphs.
    """

    skill: str                                  # owning skill bundle (legacy `agent` accepted)
    inputs: dict[str, str]                       # {name: type-name string (gap.schema)}
    outputs: dict[str, Ref]                      # {name: Ref binding to node field}
    nodes: dict[str, NodeDef]
    edges: tuple[tuple[str, str], ...]
    conditional_edges: dict[str, ConditionalEdge]
    exit: ExitDecl
    on_error: str | None = None


# ---------------------------------------------------------------------------
# Top-level workflow
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workflow:
    """Loaded v3 workflow."""

    version: int                                 # always 3
    meta: dict[str, str]
    nodes: dict[str, NodeDef]
    edges: tuple[tuple[str, str], ...]
    conditional_edges: dict[str, ConditionalEdge]
    subgraphs: dict[str, SubgraphDef]
    workflow_dir: Path                           # for resolving script paths
    requires_tools: tuple[str, ...] = ()
    """Flat tool names this workflow's SCRIPT nodes call, declared by the author.

    Tool bundles are booted from what the workflow *shows*: a ``type: tool``
    node names one directly, and a subgraph naming a skill contributes that
    bundle's ``allowed_tools``. Neither sees inside a script, and a subgraph
    labelled ``generic`` names no skill at all -- so a graph whose whole
    perception layer is its own scripts calling ``sam3.segment_text`` boots
    nothing and fails at the first call with "not found".

    That is not hypothetical. Relabelling a graph's four perception subgraphs
    ``generic`` -- correct, because they ran no bundle script -- silently
    stopped its vision body booting the segmenter it had always used. The
    dishonest labels had been doing load-bearing work.

    So an author can say it outright. This is a declaration of need, not of
    provenance: it changes which bundles boot and nothing else, and a name here
    that no script calls costs one subprocess.

    **It is no longer the only way, and should not have been.** A declaration
    only helps an author who knows the rule, and the failure states it nowhere:
    a self-learning agent that had never been told spent three iterations and
    thirty-six episodes writing fallbacks around a segmenter nobody had
    started. ``gap.runtime.tool_bundle_boot.script_tool_names`` now reads the
    graph's own Python and boots what it literally calls, so the common case
    needs no declaration at all. This stays for the names a reader cannot see:
    a tool name assembled at run time, or one reached through a helper that
    does not ship with the graph.
    """

    def begin_targets(self) -> list[str]:
        """Nodes that START fans out to."""
        return [dst for src, dst in self.edges if src == START]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_WORKFLOW_KEYS = frozenset({
    "version", "meta", "nodes", "edges", "conditional_edges", "subgraphs",
    # {"tools": ["sam3.segment_text", ...]} -- see Workflow.requires_tools.
    "requires",
})
_SUBGRAPH_KEYS = frozenset({
    "skill", "agent", "inputs", "outputs",
    "nodes", "edges", "conditional_edges", "exit", "on_error",
    # Canonical pick-and-place stage tag ("grasp"/"transport"/"place").
    # Consumed by the iter-1 mechanical-swap engine
    # (gap.refine.iter1_swap); the runtime itself ignores it.
    "stage",
})
_NODE_KEYS = frozenset({
    "type", "inputs", "streaming",
    "script", "tool",
    "ref",                                        # subgraph node
    "status", "recovery",                         # end node
})
_RECOVERY_KEYS = frozenset({"tool", "inputs"})
_LEGACY_RECOVERY_KEYS = frozenset({"service", "method"})
_COND_EDGE_KEYS = frozenset({"router_field", "mapping"})
_EXIT_KEYS = frozenset({"router_field", "success_values"})

_VALID_NODE_TYPES = frozenset({
    "tool", "script", "router", "subgraph", "end", "noop",
})

# Legacy node types that v3 used to accept. They have been retired in
# favour of the unified `type: tool` surface (see SKILL.md state-flow
# examples). The validator surfaces a precise migration message instead
# of the generic "invalid type" error so the failure is self-describing.
_LEGACY_NODE_TYPES = frozenset({"service", "skill", "policy"})


def load_workflow(path: str | Path) -> Workflow:
    """Load and parse a v3 workflow.json file.

    Performs syntactic parsing with strict key validation. Structural
    rules (reachability, edge consistency, streaming-flag/skill-contract
    matching, etc.) live in :mod:`gap.runtime.validate` and are run
    separately by the executor (and codegen pipeline).

    Raises ``WorkflowValidationError`` on syntax errors, unknown keys,
    or v2 leakage.
    """
    path = Path(path)
    if not path.exists():
        raise WorkflowValidationError(f"Workflow file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    return _parse_workflow(raw, path.parent)


def _check_keys(raw: dict, allowed: frozenset[str], location: str) -> None:
    extra = set(raw.keys()) - allowed
    if extra:
        raise WorkflowValidationError(
            f"{location} has unknown fields {sorted(extra)}; "
            f"allowed: {sorted(allowed)}"
        )


def _parse_workflow(raw: dict, workflow_dir: Path) -> Workflow:
    if not isinstance(raw, dict):
        raise WorkflowValidationError("workflow root must be a JSON object")
    _check_keys(raw, _WORKFLOW_KEYS, "workflow root")

    version = raw.get("version")
    if version != SUPPORTED_VERSION:
        raise WorkflowValidationError(
            f"unsupported workflow version {version!r}, expected {SUPPORTED_VERSION}"
        )

    meta = raw.get("meta", {})
    if not isinstance(meta, dict):
        raise WorkflowValidationError("workflow.meta must be an object")

    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, dict) or not nodes_raw:
        raise WorkflowValidationError("workflow.nodes must be a non-empty object")
    nodes = _parse_nodes(nodes_raw, "workflow.nodes")

    edges = _parse_edges(raw.get("edges", []), "workflow.edges")
    conditional_edges = _parse_conditional_edges(
        raw.get("conditional_edges", {}), "workflow.conditional_edges",
    )

    subgraphs_raw = raw.get("subgraphs", {})
    if not isinstance(subgraphs_raw, dict):
        raise WorkflowValidationError("workflow.subgraphs must be an object")
    subgraphs: dict[str, SubgraphDef] = {}
    for sg_name, sg_raw in subgraphs_raw.items():
        if not isinstance(sg_raw, dict):
            raise WorkflowValidationError(
                f"subgraph '{sg_name}' must be an object"
            )
        subgraphs[sg_name] = _parse_subgraph(sg_name, sg_raw)

    requires_tools = _parse_requires(raw.get("requires", {}))

    return Workflow(
        version=version,
        meta=meta,
        nodes=nodes,
        edges=edges,
        conditional_edges=conditional_edges,
        subgraphs=subgraphs,
        workflow_dir=workflow_dir,
        requires_tools=requires_tools,
    )


def _parse_requires(raw: object) -> tuple[str, ...]:
    """``requires: {tools: [...]}`` -- the tool names a script node calls."""
    if not raw:
        return ()
    if not isinstance(raw, dict):
        raise WorkflowValidationError("workflow.requires must be an object")
    _check_keys(raw, frozenset({"tools"}), "workflow.requires")
    tools = raw.get("tools", [])
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise WorkflowValidationError("workflow.requires.tools must be a list of strings")
    return tuple(tools)


def _parse_subgraph(name: str, raw: dict) -> SubgraphDef:
    _check_keys(raw, _SUBGRAPH_KEYS, f"subgraph '{name}'")

    skill = raw.get("skill")
    if skill is None:
        skill = raw.get("agent", "")
    if not isinstance(skill, str):
        raise WorkflowValidationError(
            f"subgraph '{name}' requires a string 'skill' or 'agent' field"
        )

    inputs_raw = raw.get("inputs", {})
    if not isinstance(inputs_raw, dict):
        raise WorkflowValidationError(
            f"subgraph '{name}'.inputs must be an object"
        )
    inputs: dict[str, str] = {}
    for in_name, in_type in inputs_raw.items():
        if not isinstance(in_type, str):
            raise WorkflowValidationError(
                f"subgraph '{name}'.inputs.{in_name} must be a type-name string"
            )
        inputs[in_name] = in_type

    outputs_raw = raw.get("outputs", {})
    if not isinstance(outputs_raw, dict):
        raise WorkflowValidationError(
            f"subgraph '{name}'.outputs must be an object"
        )
    outputs: dict[str, Ref] = {}
    for out_name, out_val in outputs_raw.items():
        if not is_ref_dict(out_val):
            raise WorkflowValidationError(
                f"subgraph '{name}'.outputs.{out_name} must be a "
                f"{{'$ref': '...'}} object binding to an internal node field"
            )
        outputs[out_name] = Ref(path=out_val["$ref"])

    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, dict) or not nodes_raw:
        raise WorkflowValidationError(
            f"subgraph '{name}'.nodes must be a non-empty object"
        )
    if RESERVED_INPUT_PSEUDOSTATE in nodes_raw:
        raise WorkflowValidationError(
            f"subgraph '{name}'.nodes cannot contain a node named "
            f"'{RESERVED_INPUT_PSEUDOSTATE}' (reserved for bound inputs)"
        )
    nodes = _parse_nodes(nodes_raw, f"subgraph '{name}'.nodes")

    edges = _parse_edges(raw.get("edges", []), f"subgraph '{name}'.edges")
    cond_edges = _parse_conditional_edges(
        raw.get("conditional_edges", {}),
        f"subgraph '{name}'.conditional_edges",
    )

    exit_raw = raw.get("exit")
    if not isinstance(exit_raw, dict):
        raise WorkflowValidationError(
            f"subgraph '{name}'.exit must be an object with router_field/success_values"
        )
    if "values" in exit_raw:
        raise WorkflowValidationError(
            f"subgraph '{name}'.exit: legacy 'values' key is no longer supported. "
            f"Use 'success_values' (list of success-path exits) and put the single "
            f"failure exit in the subgraph's sibling 'on_error' field. "
            f"Run `tools/rewrite_workflow_json.py --exit-split <path>` to migrate."
        )
    _check_keys(exit_raw, _EXIT_KEYS, f"subgraph '{name}'.exit")
    router_field = exit_raw.get("router_field")
    if router_field is not None and not isinstance(router_field, str):
        raise WorkflowValidationError(
            f"subgraph '{name}'.exit.router_field must be a string or null"
        )
    success_values_raw = exit_raw.get("success_values", [])
    if not isinstance(success_values_raw, list) or not all(
        isinstance(v, str) for v in success_values_raw
    ):
        raise WorkflowValidationError(
            f"subgraph '{name}'.exit.success_values must be a list of strings"
        )

    on_error = raw.get("on_error")
    if on_error is not None and not isinstance(on_error, str):
        raise WorkflowValidationError(
            f"subgraph '{name}'.on_error must be a string or absent"
        )

    return SubgraphDef(
        skill=skill,
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        edges=edges,
        conditional_edges=cond_edges,
        exit=ExitDecl(router_field=router_field, success_values=tuple(success_values_raw)),
        on_error=on_error,
    )


def _parse_nodes(raw: dict, location: str) -> dict[str, NodeDef]:
    nodes: dict[str, NodeDef] = {}
    for node_name, node_raw in raw.items():
        if not isinstance(node_raw, dict):
            raise WorkflowValidationError(
                f"{location}.{node_name} must be an object"
            )
        if node_name in (START, END):
            raise WorkflowValidationError(
                f"{location}.{node_name} uses reserved virtual name"
            )
        nodes[node_name] = _parse_node(node_name, node_raw, location)
    return nodes


def _parse_node(node_name: str, raw: dict, location: str) -> NodeDef:
    loc = f"{location}.{node_name}"
    # Surface the legacy-type migration message BEFORE the unknown-fields
    # check, so the user sees a precise rewrite hint instead of a generic
    # "unknown field 'service'" error.
    node_type = raw.get("type")
    if node_type in _LEGACY_NODE_TYPES:
        raise WorkflowValidationError(
            f"{loc} uses retired type {node_type!r}. "
            f"Rewrite as `type: \"tool\"` with a flat dispatch name: "
            f"`{{\"type\": \"tool\", \"tool\": \"<flat_name>\"}}`. "
            f"For connector methods, the flat name is "
            f"`robot.<method_name>` (e.g. `robot.get_observation`); "
            f"for atomic skills, use the skill name (e.g. "
            f"`run_policy`); for learned policies, use the "
            f"policy name."
        )
    _check_keys(raw, _NODE_KEYS, loc)
    if node_type not in _VALID_NODE_TYPES:
        raise WorkflowValidationError(
            f"{loc} has invalid type {node_type!r}; "
            f"expected one of {sorted(_VALID_NODE_TYPES)}"
        )

    streaming = bool(raw.get("streaming", False))
    if streaming and node_type not in ("tool", "script"):
        raise WorkflowValidationError(
            f"{loc} has streaming=true but type={node_type!r}; "
            f"streaming is only valid on tool or script nodes"
        )

    inputs_raw = raw.get("inputs", {})
    if not isinstance(inputs_raw, dict):
        raise WorkflowValidationError(f"{loc}.inputs must be an object")
    parsed_inputs: dict[str, Any] = {
        k: _parse_input_value(v) for k, v in inputs_raw.items()
    }

    if node_type == "end":
        status = raw.get("status")
        if status not in ("success", "failure"):
            raise WorkflowValidationError(
                f"{loc} (type=end) requires status 'success' or 'failure', "
                f"got {status!r}"
            )
        recovery_raw = raw.get("recovery", []) or []
        if not isinstance(recovery_raw, list):
            raise WorkflowValidationError(
                f"{loc}.recovery must be a list of tool-call objects"
            )
        recovery: list[ToolCall] = []
        for i, op in enumerate(recovery_raw):
            if not isinstance(op, dict):
                raise WorkflowValidationError(
                    f"{loc}.recovery[{i}] must be an object"
                )
            # Surface the legacy-form migration message BEFORE the
            # unknown-fields check, so the user sees a precise rewrite
            # hint instead of a generic "unknown field 'service'" error.
            if _LEGACY_RECOVERY_KEYS & set(op.keys()):
                raise WorkflowValidationError(
                    f"{loc}.recovery[{i}] uses the retired service/method "
                    f"form; recovery entries now name a tool — run the "
                    f"tool-name migration script to rewrite as "
                    f"`{{\"tool\": \"<flat_name>\", \"inputs\": {{...}}}}`"
                )
            _check_keys(op, _RECOVERY_KEYS, f"{loc}.recovery[{i}]")
            tool = op.get("tool")
            if not isinstance(tool, str) or not tool:
                raise WorkflowValidationError(
                    f"{loc}.recovery[{i}] requires a string 'tool' field"
                )
            op_inputs = op.get("inputs", {}) or {}
            if not isinstance(op_inputs, dict):
                raise WorkflowValidationError(
                    f"{loc}.recovery[{i}].inputs must be an object"
                )
            recovery.append(ToolCall(tool=tool, inputs=op_inputs))
        return NodeDef(
            type="end",
            status=status,
            recovery=tuple(recovery),
            inputs=parsed_inputs,
        )

    if node_type == "subgraph":
        ref = raw.get("ref")
        if not isinstance(ref, str) or not ref:
            raise WorkflowValidationError(
                f"{loc} (type=subgraph) requires a string 'ref' field"
            )
        return NodeDef(type="subgraph", ref=ref, inputs=parsed_inputs)

    if node_type == "router":
        router_script = raw.get("script")
        if not isinstance(router_script, str) or not router_script:
            raise WorkflowValidationError(
                f"{loc} (type=router) requires a 'script' field naming "
                f"the routing function"
            )
        return NodeDef(
            type="router", script=router_script, inputs=parsed_inputs,
        )

    if node_type == "noop":
        return NodeDef(type="noop", inputs=parsed_inputs)

    # tool / script
    node = NodeDef(
        type=node_type,
        inputs=parsed_inputs,
        streaming=streaming,
        script=raw.get("script"),
        tool=raw.get("tool"),
    )
    if node_type == "script" and node.script is None:
        raise WorkflowValidationError(
            f"{loc} (type=script) requires a 'script' field"
        )
    if node_type == "tool" and node.tool is None:
        raise WorkflowValidationError(
            f"{loc} (type=tool) requires a 'tool' field"
        )
    return node


def _parse_edges(
    raw: Any, location: str,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        raise WorkflowValidationError(
            f"{location} must be a list of [src, dst] pairs"
        )
    edges: list[tuple[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 2 \
                or not all(isinstance(x, str) for x in item):
            raise WorkflowValidationError(
                f"{location}[{i}] must be a [src, dst] pair of strings, got {item!r}"
            )
        src, dst = item[0], item[1]
        edges.append((src, dst))
    return tuple(edges)


def _parse_conditional_edges(
    raw: Any, location: str,
) -> dict[str, ConditionalEdge]:
    if not isinstance(raw, dict):
        raise WorkflowValidationError(
            f"{location} must be an object keyed by source node"
        )
    out: dict[str, ConditionalEdge] = {}
    for src, ce_raw in raw.items():
        if not isinstance(ce_raw, dict):
            raise WorkflowValidationError(
                f"{location}.{src} must be an object"
            )
        _check_keys(ce_raw, _COND_EDGE_KEYS, f"{location}.{src}")
        rf = ce_raw.get("router_field")
        if rf is not None and not isinstance(rf, str):
            raise WorkflowValidationError(
                f"{location}.{src}.router_field must be a string or null"
            )
        mapping_raw = ce_raw.get("mapping")
        if not isinstance(mapping_raw, dict) or not mapping_raw:
            raise WorkflowValidationError(
                f"{location}.{src}.mapping must be a non-empty object"
            )
        mapping: dict[str, str] = {}
        for k, v in mapping_raw.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise WorkflowValidationError(
                    f"{location}.{src}.mapping must map string→string"
                )
            mapping[k] = v
        out[src] = ConditionalEdge(router_field=rf, mapping=mapping)
    return out


def _parse_input_value(value: Any) -> Any:
    """Parse a state-input value: literals pass through; ``$ref`` → Ref.

    Recurses into lists so a literal list whose elements are refs becomes
    a list of Ref objects, which the resolver flattens element-wise.
    """
    if is_ref_dict(value):
        return Ref(path=value["$ref"])
    if isinstance(value, list):
        return [_parse_input_value(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------


def resolve_inputs(
    inputs: dict[str, Any],
    local_outputs: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a node's input dict, replacing Ref values with actual values.

    ``local_outputs`` is keyed by node name. The reserved key
    ``RESERVED_INPUT_PSEUDOSTATE`` ("in") holds the bound subgraph
    inputs. A Ref whose head names a streaming node resolves through
    ``StreamSlot.latest()`` (see ``gap.runtime.executor``).
    """
    resolved: dict[str, Any] = {}
    for name, value in inputs.items():
        resolved[name] = _resolve_value(value, local_outputs)
    return resolved


def _resolve_value(value: Any, local_outputs: dict[str, Any]) -> Any:
    if isinstance(value, Ref):
        return resolve_ref(value, local_outputs)
    if isinstance(value, list):
        return [_resolve_value(v, local_outputs) for v in value]
    return value


def resolve_ref(ref: Ref, local_outputs: dict[str, Any]) -> Any:
    """Resolve a ``{"$ref": "head.field.subfield"}`` reference.

    Walks: ``local_outputs[head]``, then dict-key / attribute / int-index
    access for each subsequent path part.

    For streaming nodes, ``local_outputs[head]`` is a ``StreamSlot``;
    its ``latest()`` is invoked transparently to retrieve a snapshot.
    """
    parts = ref.parts()
    if not parts or not parts[0]:
        raise WorkflowValidationError(f"empty $ref path {ref.path!r}")

    head = parts[0]
    if head not in local_outputs:
        raise WorkflowValidationError(
            f"$ref {ref.path!r}: unknown node {head!r} in current scope"
        )

    result = local_outputs[head]

    # Streaming-node snapshot: if the slot exposes a latest() callable,
    # invoke it to get a snapshot value before walking sub-fields.
    if hasattr(result, "latest") and callable(result.latest):
        result = result.latest()

    for i, part in enumerate(parts[1:], start=1):
        result = _walk_field(ref.path, parts[: i + 1], result, part)
    return result


def _walk_field(full_path: str, walked: list[str], result: Any, part: str) -> Any:
    """One step of field access: int index, dict key, or attribute."""
    if part.lstrip("-").isdigit():
        idx = int(part)
        if isinstance(result, (list, tuple)):
            try:
                return result[idx]
            except IndexError as exc:
                raise WorkflowValidationError(
                    f"$ref {full_path!r}: index {idx} out of range at "
                    f"{'.'.join(walked)!r}"
                ) from exc
        raise WorkflowValidationError(
            f"$ref {full_path!r}: cannot index non-sequence at "
            f"{'.'.join(walked)!r} (type {type(result).__name__})"
        )

    # Dict-key access first: graph values are plain TypedDicts, and a key
    # must never be shadowed by a same-named dict method (e.g. "items").
    if isinstance(result, dict) and part in result:
        return result[part]

    if hasattr(result, part):
        return getattr(result, part)

    raise WorkflowValidationError(
        f"$ref {full_path!r}: field {part!r} not found at "
        f"{'.'.join(walked[:-1]) or '<root>'!r} (type {type(result).__name__})"
    )
