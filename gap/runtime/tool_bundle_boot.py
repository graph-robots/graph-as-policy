"""Discover which RPC-protocol tool bundles a workflow needs and boot them.

Parallel to :mod:`gap.runtime.policy_boot`. We discover the bundle set
from two sources:

1. Direct ``tool:`` nodes in the workflow (e.g. a node with
   ``tool: "sam3.segment"`` → the ``sam3`` bundle).
2. The ``allowed_tools`` list of every skill subgraph the workflow uses.
   Script nodes call tools via ``ctx.tool_call(...)``, not as workflow
   nodes — so a workflow's `target_sg` subgraph referencing a
   ``perceiving-objects`` skill must pre-boot every bundle that
   skill's ``allowed_tools`` declares (sam3, grounding-dino, vlm, …).
3. The workflow's own ``requires.tools``. Neither of the two sources above
   sees inside a script, and a subgraph labelled ``generic`` names no skill
   to ask — so a graph whose perception layer is its own scripts had no way
   to say what it calls. See ``Workflow.requires_tools``.
4. **The graph's own Python, read.** Source 3 asks the author to know a rule
   nobody states at the point of failure, and an author who does not know it
   gets no bundle and a ``Tool 'sam3.segment_text' not found`` at the first
   call — with a message that lists what IS available and never says a bundle
   must be declared. Measured cost of exactly that, on a twelve-world
   self-learning cell: three iterations, thirty-six episodes and three model
   sessions, all spent writing fallbacks around a segmenter that was never
   started. The graph had said what it calls, in the only place it could — its
   scripts — so this reads them.

A bundle whose SKILL.md declares ``gap.serving.protocol == "stdio-msgpack"``
is booted out-of-process via :class:`ToolBundleManager`. Bundles without a
``gap.serving:`` block (or with ``in-process`` protocol) are unaffected:
their @tool registrations land in the registry via the in-process path
exactly as today.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_TOOL_CALL_ATTRS = frozenset({"tool", "tool_call"})
"""Context methods that dispatch a flat tool name: ``ctx.tool("sam3.segment_text", ...)``."""


def script_tool_names(workflow_dir: str | Path) -> set[str]:
    """Flat tool names the graph's own Python calls, read out of the source.

    Every ``*.py`` under the workflow directory, not only the files named by
    script nodes: a script may put its perception body in a helper beside it,
    and a bundle that boots for the node but not for the helper is the same
    failure one import deeper.

    Only a **string literal in first position** of a ``ctx.tool(...)`` or
    ``ctx.tool_call(...)`` call counts, which is what makes this safe to run
    over a whole directory -- a tool name in a docstring or a comment is not a
    call and cannot start a subprocess. A name computed at run time is
    invisible here by construction, and that is what ``requires.tools`` stays
    for.

    Never raises. A file that does not parse, or does not read, contributes
    nothing: this runs before a graph starts, and a graph that would otherwise
    run must not be stopped by a scanner.
    """
    found: set[str] = set()
    try:
        paths = sorted(Path(workflow_dir).rglob("*.py"))
    except OSError:
        return found
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in _TOOL_CALL_ATTRS:
                continue
            first = node.args[0]
            # ``bundle.tool``: a bare name is a connector method, never a bundle.
            if isinstance(first, ast.Constant) and isinstance(first.value, str) and "." in first.value:
                found.add(first.value)
    return found


def required_rpc_tool_bundles(
    workflow_dir: str | Path, skill_registry: Any, tool_registry: Any = None,
) -> set[str]:
    """Bundle names a workflow references whose serving.protocol is
    ``stdio-msgpack``.

    When *tool_registry* is given, a bundle every one of whose referenced
    tool names is already registered there is not required: the connector
    provided those tools in-process, and in-process wins. Simulator
    connectors (RoboSimStudio's ``ConnectorNewton``) register ``geometry.*``
    and friends directly, and booting the same-named bundle over them was a
    guaranteed name collision at graph startup — the bundle exists to serve
    the tools, not to outrank a connector that already has.
    """
    out: set[str] = set()
    if skill_registry is None:
        return out
    from gap.runtime.workflow import load_workflow

    try:
        wf = load_workflow(Path(workflow_dir) / "workflow.json")
    except Exception:
        return out

    candidate_tool_names: set[str] = set(getattr(wf, "requires_tools", ()) or ())
    from_scripts = script_tool_names(workflow_dir)
    if from_scripts - candidate_tool_names:
        logger.info(
            "tool names read from this graph's scripts: %s",
            ", ".join(sorted(from_scripts - candidate_tool_names)),
        )
    candidate_tool_names |= from_scripts
    candidate_skills: set[str] = set()
    for sg in wf.subgraphs.values():
        # Skill that owns this subgraph contributes its allowed_tools — script
        # nodes call those via ctx.tool_call(...) and aren't workflow nodes.
        skill_name = getattr(sg, "skill", None)
        if skill_name:
            candidate_skills.add(skill_name)
        for node in sg.nodes.values():
            if node.type == "tool" and node.tool:
                candidate_tool_names.add(node.tool)

    for skill_name in candidate_skills:
        try:
            info = skill_registry.get(skill_name)
        except Exception:
            continue
        for tool_name in getattr(info.meta, "allowed_tools", []) or []:
            candidate_tool_names.add(tool_name)

    by_bundle: dict[str, set[str]] = {}
    for tool_name in candidate_tool_names:
        by_bundle.setdefault(tool_name.split(".", 1)[0], set()).add(tool_name)

    for bundle, names in by_bundle.items():
        try:
            info = skill_registry.get(bundle)
        except Exception:
            continue
        serving = getattr(info.meta, "serving", None)
        if serving is None:
            continue
        if getattr(serving, "protocol", None) != "stdio-msgpack":
            continue
        if tool_registry is not None and all(n in tool_registry for n in names):
            logger.info(
                "tool bundle %r not booted: %s already provided in-process",
                bundle, ", ".join(sorted(names)),
            )
            continue
        out.add(bundle)
    return out


def boot_tool_bundles(
    workflow_dir: str | Path,
    skill_registry: Any,
    tool_registry: Any,
    *,
    startup_timeout_s: float = 60.0,
    evict_grace_s: float = 5.0,
) -> Any | None:
    """Boot every RPC tool bundle the workflow needs.

    Returns the :class:`ToolBundleManager` or ``None`` when no RPC bundle
    is referenced (the common case today — all current tool bundles run
    in-process). The caller MUST call ``manager.shutdown_all()`` in a
    ``finally`` to terminate the subprocesses.
    """
    required = required_rpc_tool_bundles(workflow_dir, skill_registry, tool_registry)
    if not required:
        return None

    from gap.runtime.tool_bundle_manager import ToolBundleManager

    manager = ToolBundleManager(
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        startup_timeout_s=startup_timeout_s,
        evict_grace_s=evict_grace_s,
    )
    manager.boot_all(required)
    return manager


__all__ = ["boot_tool_bundles", "required_rpc_tool_bundles", "script_tool_names"]
