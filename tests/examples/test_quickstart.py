"""libero_quickstart example: default-suite validation + live e2e regression.

Default suite (no markers): the two quickstart graphs load, validate
against the open-robot-skills checkout, reference only known tool names, and
reference only scripts/checkpoints that exist on disk.

The ``sim+gpu+llm``-marked test is the live end-to-end regression: real
SAM3 + Grounding DINO weights, a live VLM provider, and the LIBERO sim.
Run it explicitly::

    pytest tests/examples/test_quickstart.py -m "sim" --no-header

with ``MUJOCO_GL=egl``, a CUDA device, and a configured VLM provider
(``OPENROUTER_API_KEY``, or ``GAP_VLM_PROVIDER=vertex`` + project env).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "libero_quickstart"

GRAPHS = ["graph", "graph_planner"]

# Tools the connector registers at runtime (robot.* / sim.*). They are not
# in the static @tool registry (no env at validate time), so the known-name
# check accepts them from this explicit surface list.
CONNECTOR_TOOLS = frozenset({
    "robot.close_gripper", "robot.execute_trajectory", "robot.get_camera_pose",
    "robot.get_ee_pose", "robot.get_gripper", "robot.get_gripper_pose",
    "robot.get_observation", "robot.go_home", "robot.go_to_pose",
    "robot.go_to_pose_cartesian", "robot.move_to_joints", "robot.open_gripper",
    "robot.solve_ik",
    "sim.apply_policy_action", "sim.check_success", "sim.enable_video",
    "sim.reset", "sim.save_video", "sim.step",
})


def _skills_root() -> Path:
    """The sibling open-robot-skills checkout (skipped when not present)."""
    candidate = REPO_ROOT.parent / "open-robot-skills"
    if not (candidate / "skills").is_dir():
        pytest.skip(f"open-robot-skills checkout not found at {candidate}")
    return candidate


def _graph_tools(graph_dir: Path) -> set[str]:
    raw = json.loads((graph_dir / "workflow.json").read_text())
    tools: set[str] = set()

    def _walk(obj):
        if isinstance(obj, dict):
            t = obj.get("tool")
            if isinstance(t, str):
                tools.add(t)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(raw)
    return tools


@pytest.mark.parametrize("graph", GRAPHS)
def test_quickstart_graph_validates(graph: str) -> None:
    """Both quickstart graphs load and validate with the skills registry."""
    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow
    from gap.skills import load_skills

    wf = load_workflow(EXAMPLE_DIR / graph / "workflow.json")
    skill_registry = load_skills(_skills_root())
    issues = validate_workflow(wf, skill_registry=skill_registry)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], [str(i) for i in errors]


@pytest.fixture(scope="module")
def bundle_tool_registry():
    """A ToolRegistry holding every open-robot-skills bundle @tool.

    The @tool decorator pushes onto a process-global pending queue only
    at module import, and any earlier test that builds a registry drains
    that queue — so this fixture re-imports each bundle's tools module
    (re-firing the decorators) before draining into a fresh registry.
    The queue entries that belonged to other tests are restored
    afterwards so suites asserting queue membership (e.g.
    tests/skills/test_loader.py) are unaffected.
    """
    import importlib

    from gap_core.tools import ToolRegistry
    from gap_core.tools import _registry as tool_registry_mod

    from gap.skills import load_skills

    skills = load_skills(_skills_root())
    snapshot = list(tool_registry_mod._PENDING_TOOLS)
    tool_registry_mod._PENDING_TOOLS.clear()
    for info in skills.list_skills():
        if info.tools_module is not None:
            importlib.reload(info.tools_module)
    reg = ToolRegistry()
    reg.discover_pending()
    # Bundles whose serving.protocol is stdio-msgpack run out-of-process —
    # their @tool decorators never fire in this process. Register a stub
    # descriptor for every tool name they declare in SKILL.md gap.tools so
    # the workflow's tool-name validator finds them.
    for info in skills.list_skills():
        serving = getattr(info.meta, "serving", None)
        if serving is None or getattr(serving, "protocol", None) != "stdio-msgpack":
            continue
        for tool_name, summary in (info.meta.tools or {}).items():
            if tool_name in reg:
                continue
            reg.register_rpc(tool_name, client=None, summary=summary)
    tool_registry_mod._PENDING_TOOLS.extend(snapshot)
    return reg


@pytest.mark.parametrize("graph", GRAPHS)
def test_quickstart_tool_names_known(graph: str, bundle_tool_registry) -> None:
    """Every `tool:` reference resolves to a bundle tool or connector tool."""
    unknown = {
        t for t in _graph_tools(EXAMPLE_DIR / graph)
        if t not in bundle_tool_registry and t not in CONNECTOR_TOOLS
    }
    assert unknown == set(), f"unknown tool names in {graph}: {sorted(unknown)}"


@pytest.mark.parametrize("graph", GRAPHS)
def test_quickstart_scripts_exist(graph: str) -> None:
    """Every `script:` path in the workflow exists in the example dir."""
    graph_dir = EXAMPLE_DIR / graph
    raw = json.loads((graph_dir / "workflow.json").read_text())
    missing: list[str] = []

    def _walk(obj):
        if isinstance(obj, dict):
            s = obj.get("script")
            if isinstance(s, str) and not (graph_dir / s).is_file():
                missing.append(s)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(raw)
    assert missing == [], f"missing scripts in {graph}: {missing}"


def test_quickstart_no_legacy_tool_names() -> None:
    """The migration script reports the example fully migrated."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import migrate_tool_names as mig
    finally:
        sys.path.pop(0)

    hits: list[str] = []
    for graph in GRAPHS:
        graph_dir = EXAMPLE_DIR / graph
        # Only the authored artifacts: workflow.json + scripts/ +
        # checkpoints/ (a stray runtime trace dir must not fail the check).
        targets = [graph_dir / "workflow.json"]
        for sub in ("scripts", "checkpoints"):
            targets.extend(sorted((graph_dir / sub).rglob("*.py")))
        for path in targets:
            if path.is_file():
                _, file_hits = mig.migrate_file(path, check=True)
                hits.extend(f"{path}: {h}" for h in file_hits)
    assert hits == [], hits


def test_quickstart_checkpoint_module_loads() -> None:
    """The grasp_sg checkpoints sidecar loads and exposes target_held."""
    from gap.runtime.verify import StubWorld, evaluate_checkpoint, load_checkpoints

    cps = load_checkpoints(EXAMPLE_DIR / "graph" / "checkpoints" / "grasp_sg.py")
    assert [c.name for c in cps] == ["target_held"]
    assert cps[0].validate is True

    # Dry-run against a stub world: no contacts -> the predicate is False
    # but must evaluate without raising (BodyNotFoundError etc.).
    world = StubWorld(body_names=["alphabet_soup", "basket", "table_top"])
    result = evaluate_checkpoint(cps[0], world)
    assert result.eval_error is None
    assert result.passed is False


# ---------------------------------------------------------------------------
# Live end-to-end regression (real models + sim + VLM)
# ---------------------------------------------------------------------------


@pytest.mark.sim
@pytest.mark.gpu
@pytest.mark.llm
def test_quickstart_live_e2e_seed1(tmp_path) -> None:
    """One full pick-the-soup-can-into-basket rollout on seed 1.

    Requires: [libero] extra + EGL, the sam3 / grounding-dino / geometry bundles' weights on a CUDA
    device, and a live VLM provider. ~2-4 minutes.
    """
    pytest.importorskip("gap.envs.registry")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    pytest.importorskip("sam3")
    pytest.importorskip("transformers")

    import gap.connector
    from gap.runtime.execute import execute

    conn = gap.connector.sim("libero", task="libero_object_all_variance/0", seed=1)
    try:
        conn.reset()
        result = execute(
            str(EXAMPLE_DIR / "graph"),
            conn,
            skills=str(_skills_root()),
            checkpoints="warn",
            # Explicit trace dir: execute() defaults trace output INTO the
            # workflow directory, which would pollute the checked-in example.
            trace_dir=str(tmp_path / "trace"),
        )
        assert result.error is None, result.error
        assert result.success, f"exit={result.exit_status}"
        # The grasp checkpoint fired and passed.
        held = [c for c in result.checkpoint_results if c.name == "target_held"]
        assert held and held[0].passed, held
        # Ground-truth task success.
        ok, _reward = conn.check_success()
        assert ok, "sim reports task not completed"
    finally:
        conn.close()
