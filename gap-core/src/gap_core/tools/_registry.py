"""ToolRegistry, ToolDescriptor, and the @tool decorator.

The registry is the single point of truth for which tools exist. Discovery
walks a directory of ``@tool``-decorated modules and registers a
``ToolDescriptor`` for each declaration; connectors register their
``robot.*`` / ``sim.*`` tools explicitly via
:meth:`ToolRegistry.register_callable`.

The descriptor's tool dispatches through :meth:`ToolRegistry.invoke`, which
delegates to the in-process Python adapter. The LLM never sees the dispatch
mechanics — it sees a flat catalog of typed tools.
"""

from __future__ import annotations

import difflib
import importlib.util
import inspect
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from gap_core.errors import ToolArgumentError
from gap_core.tools.schema import UnitSchema, extract_schema

logger = logging.getLogger(__name__)


def _with_suggestions(unknown: list[str], accepted: set[str]) -> str:
    """``'arm' (did you mean 'arm_id'?)`` -- name the fix, not just the fault.

    Most contract mistakes are near misses of a real parameter, because the
    surface is not uniform: ``motion.plan_linear`` takes ``end`` while
    ``motion.plan_joint`` takes ``pose``, and ``sim.check_grasp`` takes ``arm``
    while everything else takes ``arm_id``. A caller regularizing one to the
    other is reasoning correctly about an irregular surface, and the error is
    the only place that can say so.
    """
    parts = []
    for u in unknown:
        close = difflib.get_close_matches(u, sorted(accepted), n=1, cutoff=0.6)
        parts.append(f"{u!r} (did you mean {close[0]!r}?)" if close else repr(u))
    return ", ".join(parts)

ToolScope = Literal["runtime", "codegen"]
ToolTransport = Literal["python", "rpc"]

#: Name prefixes reserved for connector-registered tools. ``@tool``
#: declarations and bundle discovery may not claim these — the connector
#: registers them explicitly via :meth:`ToolRegistry.register_callable`.
RESERVED_TOOL_PREFIXES: tuple[str, ...] = ("robot.", "sim.")


class Adapter(Protocol):
    """Tool dispatch adapter.

    Adapters are stateful: they hold whatever callable map is needed to
    dispatch a tool. The Python adapter holds a name → callable map.
    """

    transport: ToolTransport

    def invoke(self, name: str, ctx: Any | None, **kwargs: Any) -> Any: ...


@dataclass
class ToolDescriptor:
    """Registry entry for one tool — same shape regardless of origin."""

    name: str                                 # canonical name; what the LLM emits
    summary: str                              # one-line description for prompts
    schema: UnitSchema                        # introspected I/O for the LLM
    transport: ToolTransport = "python"       # which adapter dispatches
    scope: ToolScope = "runtime"              # runtime vs codegen
    tags: tuple[str, ...] = ()                # for harness filtering + guard limits
    metadata: dict[str, Any] = field(default_factory=dict)
    """Adapter-specific metadata: ``{module: str, qualname: str,
    requires_ctx: bool}``. ``requires_ctx`` indicates ctx-injection."""


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

# Module-level pending registrations. The decorator stashes function refs
# here; ToolRegistry.discover_pending() drains them into the registry.
_PENDING_TOOLS: list[dict[str, Any]] = []
# Everything ever drained in this process — replayed into every registry
# that calls discover_pending(), so registries built after the first drain
# (fresh connectors, the sequential benchmark path) still get bundle tools.
_DRAINED_TOOLS: list[dict[str, Any]] = []


def tool(
    *,
    name: str,
    summary: str,
    scope: ToolScope = "runtime",
    tags: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a Python callable as a tool.

    Args:
        name: Canonical tool name. Must be unique across the registry.
            Convention: ``<namespace>.<verb>`` (e.g. ``geometry.iou``).
            The ``robot.*`` and ``sim.*`` prefixes are reserved for
            connector-registered tools and raise ``ValueError`` here.
        summary: One-line description shown to the LLM in the catalog.
        scope: ``runtime`` for tools the LLM may call from a workflow
            ``type: tool`` state; ``codegen`` for tools bound to the
            codegen LLM via the SDK ``tools=`` parameter.
        tags: Tag tuple for harness filtering and guard classification
            (perception, planning, sim_step, ...).

    The decorated function's signature is introspected by
    :func:`gap.tools.schema.extract_schema` (same machinery as skills) to
    build a ``UnitSchema``. The registry converts this to JSON Schema for
    the SDK's ``tools=`` parameter.

    For ``scope="codegen"`` tools, the runtime injects a ``CodegenContext``
    as the first positional argument iff the function declares a ``ctx``
    parameter. Pure read-only meta-tools skip injection.
    """
    if name.startswith(RESERVED_TOOL_PREFIXES):
        raise ValueError(
            f"@tool name {name!r} uses a reserved connector prefix "
            f"{RESERVED_TOOL_PREFIXES}; connectors register these via "
            f"ToolRegistry.register_callable()"
        )

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        _PENDING_TOOLS.append({
            "name": name,
            "summary": summary,
            "scope": scope,
            "tags": tuple(tags),
            "fn": fn,
        })
        return fn
    return _wrap


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class PythonAdapter:
    """Dispatch in-process Python tool callables."""

    transport: ToolTransport = "python"

    def __init__(self) -> None:
        self._callables: dict[str, Callable[..., Any]] = {}
        self._requires_ctx: dict[str, bool] = {}

    def register(self, name: str, fn: Callable[..., Any], *, requires_ctx: bool) -> None:
        self._callables[name] = fn
        self._requires_ctx[name] = requires_ctx

    def invoke(self, name: str, ctx: Any | None, **kwargs: Any) -> Any:
        """Call a registered tool, refusing arguments it does not declare.

        This used to filter unknown keys out and call anyway. That is the
        worst available behaviour for a code-writing caller: a policy that
        asked ``robot.grasp_frame(yaw=-0.105)`` when the parameter is
        ``close_heading_deg`` got the *default* frame back with a status of
        ok, and every downstream symptom pointed somewhere else. An argument
        the tool never declared is a contract error at the call site; say so
        there, with the accepted names, rather than silently substituting a
        default and letting the mistake surface as physics.
        """
        fn = self._callables.get(name)
        if fn is None:
            raise KeyError(f"Python tool {name!r} not registered with this adapter")
        sig = inspect.signature(fn)
        parameters = sig.parameters
        # The ctx-injected parameter is passed positionally and is not always
        # spelled "ctx" -- `register_callable(..., requires_ctx=True)` takes
        # whatever the first parameter is called. Identify it by position, or
        # nothing named `conn` reads as a caller-supplied argument.
        injected = (
            next(iter(parameters), None) if self._requires_ctx.get(name)
            else ("ctx" if "ctx" in parameters else None)
        )
        # A tool that declares **kwargs has opted into arbitrary keys.
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            accepted_kwargs = dict(kwargs)
        else:
            accepted = set(parameters) - ({injected} if injected else set())
            unknown = sorted(set(kwargs) - accepted)
            if unknown:
                raise ToolArgumentError(
                    name,
                    f"does not accept {_with_suggestions(unknown, accepted)}; "
                    f"its parameters are {', '.join(sorted(accepted)) or '(none)'}",
                )
            # A missing required argument reaches the caller as a TypeError
            # naming the bound *method* -- "SimTools.check_grasp() missing 1
            # required positional argument: 'object_name'" -- which names a
            # class the policy author has never seen and cannot look up. Raise
            # the same error the unknown-argument case raises, in terms of the
            # tool, so both halves of the same mistake read alike.
            missing = [
                pname for pname, param in parameters.items()
                if pname != injected
                and pname not in kwargs
                and param.default is inspect.Parameter.empty
                and param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
            if missing:
                raise ToolArgumentError(
                    name,
                    f"requires {', '.join(repr(m) for m in missing)}; "
                    f"its parameters are {', '.join(sorted(accepted)) or '(none)'}",
                )
            accepted_kwargs = kwargs
        if self._requires_ctx[name]:
            return fn(ctx, **accepted_kwargs)
        return fn(**accepted_kwargs)


class RpcAdapter:
    """Dispatch tools that live in a per-bundle subprocess via stdio msgpack.

    Holds a ``{tool_name: ToolClient}`` map. The runtime's
    ``tool_bundle_manager`` constructs the ToolClient (one subprocess per
    bundle) and registers each tool the bundle exports through here. On
    invoke, this adapter just delegates to the client's ``call`` method —
    no schema introspection, no ctx injection (the RPC path is for
    ctx-free tool functions only; see ``gap_core/rpc/server.py``).
    """

    transport: ToolTransport = "rpc"

    def __init__(self) -> None:
        # Forward declaration to avoid the import cycle at module load time
        # (gap_core.rpc.client imports nothing from this module, but the
        # symmetry is still nice).
        self._clients: dict[str, Any] = {}  # tool_name → ToolClient

    def register(self, name: str, client: Any) -> None:
        self._clients[name] = client

    def invoke(self, name: str, ctx: Any | None, **kwargs: Any) -> Any:
        client = self._clients.get(name)
        if client is None:
            raise KeyError(f"RPC tool {name!r} not registered with this adapter")
        # ctx is intentionally dropped — the RPC path is for ctx-free tools.
        # Tools that need ctx must declare `protocol: in-process` in SKILL.md
        # and stay in gap-runtime's venv.
        return client.call(name, **kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_DEFAULT_PYTHON_TOOLS_DIR = Path(__file__).parent


class ToolRegistry:
    """Flat registry of tools.

    Public API mirrors what the harness needs: ``discover``, ``get``,
    ``__contains__``, ``runtime_tools()``, ``codegen_tools()``, ``invoke()``,
    plus ``register_callable()`` for connector-owned ``robot.*`` / ``sim.*``
    tools.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        self.python_adapter = PythonAdapter()
        self.rpc_adapter = RpcAdapter()

    # --- discovery ---

    def discover(self, *, python_tools_dir: Path | None = None) -> None:
        """Walk a directory of @tool-decorated modules and register tools.

        Args:
            python_tools_dir: Directory holding @tool-decorated modules.
                Defaults to ``gap/tools/``.
        """
        python_tools_dir = python_tools_dir or _DEFAULT_PYTHON_TOOLS_DIR
        self._discover_python(python_tools_dir)

    def discover_pending(
        self, *, catalog: list[dict[str, Any]] | None = None,
    ) -> None:
        """Drain pending registrations (the @tool decorator pushes there).

        Called automatically at the end of :meth:`discover`; call it
        directly when @tool modules were imported by other means (e.g. a
        skills-bundle loader exec'ing tool modules).

        The @tool decorator pushes onto the process-global pending queue
        exactly once per module import (imports are idempotent), but
        ToolRegistry instances are per-connector — draining the queue into
        only the *first* registry would leave every later registry (the
        second ``execute()`` in a benchmark loop, a fresh connector after
        ``close()``) without any bundle tools. Callers that build repeated
        registries pass a persistent ``catalog`` list: drained entries are
        appended to it, and entries already in it are (idempotently)
        re-registered into *this* registry first.
        """
        if catalog is not None:
            for entry in catalog:
                self._register_python(
                    entry["name"], entry["summary"],
                    entry["scope"], entry["tags"], entry["fn"],
                )
        # Replay the process-global drained catalog: the pending queue is
        # one-shot per import, so a registry built AFTER another registry
        # drained it (the in-process sequential benchmark path — codegen
        # loads the bundles, then worker_setup builds a fresh registry in
        # the SAME process) would otherwise silently miss every bundle
        # tool. Entries already present (e.g. via the caller's catalog)
        # are skipped.
        for entry in _DRAINED_TOOLS:
            if entry["name"] in self:
                continue
            self._register_python(
                entry["name"], entry["summary"],
                entry["scope"], entry["tags"], entry["fn"],
            )
        while _PENDING_TOOLS:
            entry = _PENDING_TOOLS.pop(0)
            _DRAINED_TOOLS.append(entry)
            self._register_python(
                entry["name"], entry["summary"],
                entry["scope"], entry["tags"], entry["fn"],
            )
            if catalog is not None:
                catalog.append(entry)

    def register_callable(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        summary: str,
        tags: tuple[str, ...] = (),
        requires_ctx: bool | None = None,
        scope: ToolScope = "runtime",
    ) -> None:
        """Register a plain callable under an explicit name.

        The connector's entry point for ``robot.*`` / ``sim.*`` tools — the
        only path allowed to claim those reserved prefixes. Unlike module
        re-discovery there is no idempotent escape: re-registering an
        existing name always raises ``ValueError``.

        Args:
            name: Canonical tool name (reserved prefixes allowed here).
            fn: The callable to dispatch.
            summary: One-line description shown to the LLM in the catalog.
            tags: Tag tuple for harness filtering and guard classification.
            requires_ctx: Whether to inject the NodeContext as the first
                positional argument. ``None`` (default) auto-detects from a
                ``ctx`` parameter in the signature.
        """
        if name in self._tools:
            raise ValueError(f"Tool name collision: {name!r} already registered")
        self._register_python(
            name, summary, scope, tuple(tags), fn,
            requires_ctx=requires_ctx, allow_reserved=True,
        )

    def _discover_python(self, root: Path) -> None:
        if not root.is_dir():
            return
        for py in sorted(root.glob("*.py")):
            if py.name.startswith("_"):
                continue
            if py.stem == "__init__":
                continue
            self._import_python_tools_module(py, root)
        self.discover_pending()

    def _import_python_tools_module(self, path: Path, root: Path) -> None:
        rel = path.relative_to(root.parent)
        module_name = ".".join(rel.with_suffix("").parts)
        if not module_name.startswith("gap."):
            module_name = f"gap.{module_name}"
        if module_name in sys.modules:
            return
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("Cannot import Python tools module: %s", path)
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            logger.warning("Failed to import %s", path, exc_info=True)
            del sys.modules[module_name]

    def _register_python(
        self,
        name: str,
        summary: str,
        scope: ToolScope,
        tags: tuple[str, ...],
        fn: Callable[..., Any],
        *,
        requires_ctx: bool | None = None,
        allow_reserved: bool = False,
    ) -> None:
        if not allow_reserved and name.startswith(RESERVED_TOOL_PREFIXES):
            raise ValueError(
                f"Tool name {name!r} uses a reserved connector prefix "
                f"{RESERVED_TOOL_PREFIXES}; register it via "
                f"ToolRegistry.register_callable()"
            )
        if name in self._tools:
            existing = self._tools[name]
            if existing.transport == "python" and existing.metadata.get("qualname") == fn.__qualname__:
                return  # idempotent re-import
            raise ValueError(f"Tool name collision: {name!r} already registered")

        # Build a fake module for extract_schema: it expects ``module.run`` and
        # ``module._meta``. Wrap the function so the same machinery applies.
        from types import SimpleNamespace
        wrapper = SimpleNamespace(
            run=fn,
            _meta=None,
            __doc__=fn.__doc__ or summary,
            __name__=name,
            __dict__={},
        )
        wrapper.__dict__.update(getattr(sys.modules.get(fn.__module__, None), "__dict__", {}))
        try:
            schema = extract_schema(wrapper)
        except Exception as e:
            logger.warning("Failed to extract schema for tool %r: %s", name, e)
            schema = UnitSchema(name=name, description=summary)

        if requires_ctx is None:
            sig = inspect.signature(fn)
            requires_ctx = "ctx" in sig.parameters
        self.python_adapter.register(name, fn, requires_ctx=requires_ctx)
        self._tools[name] = ToolDescriptor(
            name=name,
            summary=summary,
            schema=schema,
            transport="python",
            scope=scope,
            tags=tuple(tags),
            metadata={
                "module": fn.__module__,
                "qualname": fn.__qualname__,
                "requires_ctx": requires_ctx,
            },
        )
        logger.debug("Registered python tool %r (scope=%s)", name, scope)

    # --- accessors ---

    def get(self, name: str) -> ToolDescriptor:
        if name not in self._tools:
            available = ", ".join(sorted(self._tools.keys()))
            raise KeyError(
                f"Tool {name!r} not found (available: {available[:200]}...)"
                f"{self._bundle_hint(name)}"
            )
        return self._tools[name]

    def _bundle_hint(self, name: str) -> str:
        """Why a whole family of tools can be missing, said at the point of failure.

        A dotted name whose prefix matches nothing here is most often a tool
        bundle that never booted rather than a typo: bundles are started per
        graph, from what the graph shows and from what its scripts call, so a
        name built at run time or reached through a helper outside the graph
        directory starts nothing. Listing what IS available cannot convey that
        -- a reader sees a catalogue missing one family and concludes the
        family is unavailable here. One self-learning agent concluded exactly
        that and spent three iterations writing fallbacks around a segmenter it
        could have had for one line.
        """
        prefix, dot, _ = name.partition(".")
        if not dot or any(tool.startswith(f"{prefix}.") for tool in self._tools):
            return ""
        return (
            f". No {prefix!r} tool is registered at all, so this is most likely a bundle "
            f"that was not booted rather than a wrong name: a graph boots bundles from its "
            f"tool nodes, the skills its subgraphs name, the tool names its own scripts "
            f"literally call, and its workflow-level 'requires': {{'tools': ['{name}']}}"
        )

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def runtime_tools(self) -> dict[str, ToolDescriptor]:
        return {n: d for n, d in self._tools.items() if d.scope == "runtime"}

    def codegen_tools(self) -> dict[str, ToolDescriptor]:
        return {n: d for n, d in self._tools.items() if d.scope == "codegen"}

    def by_transport(self, transport: ToolTransport) -> dict[str, ToolDescriptor]:
        return {n: d for n, d in self._tools.items() if d.transport == transport}

    # --- dispatch ---

    def invoke(self, name: str, ctx: Any | None = None, **kwargs: Any) -> Any:
        descriptor = self.get(name)
        if descriptor.transport == "python":
            return self.python_adapter.invoke(name, ctx, **kwargs)
        if descriptor.transport == "rpc":
            return self.rpc_adapter.invoke(name, ctx, **kwargs)
        raise ValueError(f"unknown transport {descriptor.transport!r}")

    # ------------------------------------------------------------------
    # RPC registration (gap.runtime.tool_bundle_manager calls this after
    # spawning a per-bundle subprocess and reading its catalog)
    # ------------------------------------------------------------------

    def register_rpc(
        self,
        name: str,
        client: Any,
        *,
        summary: str = "",
        schema: UnitSchema | None = None,
        tags: tuple[str, ...] = (),
        scope: ToolScope = "runtime",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register an RPC-routed tool fronted by ``client`` (a ToolClient).

        Unlike :meth:`register_callable`, no Python signature exists on the
        gap-runtime side — the schema comes from the bundle server's
        startup catalog frame. The descriptor's ``transport`` is set to
        ``"rpc"`` so :meth:`invoke` dispatches through :class:`RpcAdapter`.
        """
        if name in self._tools:
            raise ValueError(f"Tool name collision: {name!r} already registered")
        self.rpc_adapter.register(name, client)
        self._tools[name] = ToolDescriptor(
            name=name,
            summary=summary,
            schema=schema or UnitSchema(name=name, description=summary),
            transport="rpc",
            scope=scope,
            tags=tuple(tags),
            metadata={
                "bundle": getattr(client, "bundle_name", ""),
                **(metadata or {}),
            },
        )
        logger.debug("Registered rpc tool %r (bundle=%s)", name,
                     getattr(client, "bundle_name", "?"))


# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------


def default_tool_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with everything under gap/tools/."""
    reg = ToolRegistry()
    reg.discover()
    return reg
