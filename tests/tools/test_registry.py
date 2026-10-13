"""Unit tests for gap.tools — @tool decorator, ToolRegistry, dispatch."""

# NOTE: no `from __future__ import annotations` — extract_schema resolves the
# TypedDict return annotations defined in this module.

from typing import TypedDict

import pytest
from gap_core.errors import ToolArgumentError
from gap_core.tools import RESERVED_TOOL_PREFIXES, ToolRegistry, tool
from gap_core.tools._registry import _PENDING_TOOLS


class IouOutput(TypedDict):
    iou: float


class SumOutput(TypedDict):
    sum: int


class EchoOutput(TypedDict):
    message: str


@pytest.fixture(autouse=True)
def _clean_pending():
    """Isolate the module-level pending-tool queue between tests."""
    _PENDING_TOOLS.clear()
    yield
    _PENDING_TOOLS.clear()


# --- @tool registration + schema extraction ---


def test_tool_registration_and_schema():
    @tool(name="geometry.iou", summary="IoU of two boxes.", tags=("geometry",))
    def geometry_iou(box_a: list[float], box_b: list[float], threshold: float = 0.5) -> IouOutput:
        return {"iou": 1.0}

    reg = ToolRegistry()
    reg.discover_pending()

    assert "geometry.iou" in reg
    assert len(reg) == 1
    d = reg.get("geometry.iou")
    assert d.summary == "IoU of two boxes."
    assert d.scope == "runtime"
    assert d.tags == ("geometry",)
    assert d.metadata["requires_ctx"] is False

    # Inputs from the signature (ctx excluded, defaults captured).
    assert set(d.schema.inputs) == {"box_a", "box_b", "threshold"}
    assert d.schema.inputs["box_a"].required is True
    assert d.schema.inputs["box_a"].type_str == "list[float]"
    assert d.schema.inputs["threshold"].required is False
    assert d.schema.inputs["threshold"].default == 0.5

    # Outputs from the TypedDict return annotation.
    assert set(d.schema.outputs) == {"iou"}
    assert d.schema.outputs["iou"].type_str == "float"

    # The drain consumed the pending queue.
    assert _PENDING_TOOLS == []


def test_runtime_and_codegen_scopes():
    @tool(name="scoped.runtime_tool", summary="r")
    def runtime_tool() -> SumOutput:
        return {"sum": 0}

    @tool(name="scoped.codegen_tool", summary="c", scope="codegen")
    def codegen_tool() -> SumOutput:
        return {"sum": 0}

    reg = ToolRegistry()
    reg.discover_pending()
    assert set(reg.runtime_tools()) == {"scoped.runtime_tool"}
    assert set(reg.codegen_tools()) == {"scoped.codegen_tool"}


# --- collisions and reserved prefixes ---


def test_name_collision_raises():
    @tool(name="dup.tool", summary="first")
    def first() -> SumOutput:
        return {"sum": 1}

    @tool(name="dup.tool", summary="second")
    def second() -> SumOutput:
        return {"sum": 2}

    reg = ToolRegistry()
    with pytest.raises(ValueError, match="collision"):
        reg.discover_pending()


def test_reserved_prefixes_rejected_from_tool_decorator():
    for name in ("robot.move_to_pose", "sim.step"):
        with pytest.raises(ValueError, match="reserved"):
            tool(name=name, summary="nope")
    assert _PENDING_TOOLS == []


def test_register_callable_allows_reserved_prefix():
    reg = ToolRegistry()

    def move_to_pose(pose: list[float]) -> SumOutput:
        return {"sum": len(pose)}

    reg.register_callable(
        "robot.move_to_pose", move_to_pose, summary="Move the EEF.", tags=("planning",),
    )
    d = reg.get("robot.move_to_pose")
    assert d.tags == ("planning",)
    assert d.schema.inputs["pose"].type_str == "list[float]"
    assert reg.invoke("robot.move_to_pose", pose=[1.0, 2.0, 3.0]) == {"sum": 3}
    assert RESERVED_TOOL_PREFIXES == ("robot.", "sim.")


def test_register_callable_collision_raises_unless_default():
    reg = ToolRegistry()

    def grip() -> SumOutput:
        return {"sum": 0}

    reg.register_callable("robot.grip", grip, summary="g")
    with pytest.raises(ValueError, match="collision"):
        reg.register_callable("robot.grip", grip, summary="g again")


def test_register_callable_replaces_a_default():
    reg = ToolRegistry()

    def generic() -> SumOutput:
        return {"sum": 1}

    def specific() -> SumOutput:
        return {"sum": 2}

    reg.register_callable("robot.grip", generic, summary="generic", default=True)
    reg.register_callable("robot.grip", specific, summary="specific")
    assert reg.get("robot.grip").summary == "specific"
    assert reg.invoke("robot.grip") == {"sum": 2}
    with pytest.raises(ValueError, match="collision"):  # the replacement is not a default
        reg.register_callable("robot.grip", generic, summary="again")


# --- dispatch ---


def test_dispatch_refuses_unknown_kwargs():
    """An argument the tool does not declare is an error, not a shrug.

    Filtering it away used to look harmless. It is not: the tool then runs on
    its *defaults* and reports success, so a caller that misremembered a
    parameter name gets a plausible wrong answer instead of a correction.
    """
    @tool(name="math.add", summary="Add two ints.")
    def add(a: int, b: int) -> SumOutput:
        return {"sum": a + b}

    reg = ToolRegistry()
    reg.discover_pending()
    assert reg.invoke("math.add", a=1, b=2) == {"sum": 3}
    with pytest.raises(ToolArgumentError) as excinfo:
        reg.invoke("math.add", a=1, b=2, extraneous="ignored")
    # The message has to carry the accepted names, or the caller cannot fix it.
    assert "extraneous" in str(excinfo.value)
    assert "a, b" in str(excinfo.value)


def test_requires_ctx_injection():
    seen = {}

    @tool(name="ctxy.echo", summary="Echo via ctx.")
    def echo(ctx, message: str) -> EchoOutput:
        seen["ctx"] = ctx
        return {"message": message}

    reg = ToolRegistry()
    reg.discover_pending()
    d = reg.get("ctxy.echo")
    assert d.metadata["requires_ctx"] is True
    assert "ctx" not in d.schema.inputs

    sentinel = object()
    assert reg.invoke("ctxy.echo", ctx=sentinel, message="hi") == {"message": "hi"}
    assert seen["ctx"] is sentinel


def test_register_callable_explicit_requires_ctx():
    seen = {}

    def step(conn, n: int) -> SumOutput:
        seen["conn"] = conn
        return {"sum": n}

    reg = ToolRegistry()
    # First param is named ``conn``, not ``ctx`` — auto-detect would say
    # False; the explicit override forces injection.
    reg.register_callable("sim.step", step, summary="Step.", requires_ctx=True)
    sentinel = object()
    assert reg.invoke("sim.step", ctx=sentinel, n=4) == {"sum": 4}
    assert seen["conn"] is sentinel


def test_get_unknown_tool_raises_keyerror():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.get("no.such_tool")
