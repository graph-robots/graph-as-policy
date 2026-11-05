# Build a Graph

Author a complete pick-and-place workflow in Python with `gap.builder` —
4 subgraphs, a ground-truth checkpoint, recovery actions, and an optional
`--execute` that runs it on the LIBERO sim.

:::{note} Requirements
Building and validating runs without the sim (`uv sync`, ~1 min). The optional
`--execute` step has the same requirements as the
[LIBERO quickstart](libero-quickstart.md): `uv sync && uv run gap skills install sam3 grounding-dino geometry`,
a GPU, and a VLM credential.
:::

LLM generation is one producer of graphs, not the only one. This example
builds the complete quickstart-style pick-and-place by hand — and **the
builder is the same artifact pipeline as `gap generate`**: it emits the
identical workflow directory (`workflow.json` + `scripts/` +
`checkpoints/`), passes through the same parser and structural validation
as agent output, and runs with the same `gap run` / `gap.execute`. Source:
[examples/build_a_graph](gh-engine:examples/build_a_graph).

## Build it

```bash
uv run python examples/build_a_graph/build_graph.py --out my_graph
```

This prints the validation report and leaves a runnable artifact:

```text
my_graph/
├── workflow.json              # the v3 graph (4 subgraphs + done/abort ends)
├── scripts/                   # canonical bundle scripts, copied verbatim
│   ├── perceive_dino_vlm.py   #   from the open-robot-skills checkout — exactly the
│   ├── compute_align_pose.py  #   per-workflow copies `gap generate` emits
│   ├── compute_drop_pose.py
│   ├── waypoint_move.py
│   └── descend_release.py
└── checkpoints/
    └── grasp_sg.py            # ground-truth `target_held` postcondition
```

All flags:

| Flag | Default | Meaning |
|---|---|---|
| `--out DIR` | `my_graph` | Output workflow directory |
| `--skills PATH` | auto-discovered | open-robot-skills checkout (`$GAP_SKILLS_PATH` or the sibling checkout) |
| `--target` | `"blue and yellow alphabet soup can"` | Perception prompt for the object to pick |
| `--container` | `"basket"` | Perception prompt for the destination container |
| `--target-body` | `"alphabet soup"` | Sim body name the grasp checkpoint verifies |
| `--execute` | off | After building, run the graph on the LIBERO sim |
| `--task SUITE/ID` | `libero_object_all_variance/0` | Sim task for `--execute` |

## What gets built

The graph is real, not a toy — it mirrors the
[libero_quickstart](libero-quickstart.md) structure and runs on the same
task (`libero_object_all_variance/0`):

```text
START → target → container → grasp → transport → done
           ↘ not_found       ↘ failed   ↘ blocked → abort (open gripper, go home)
```

- **target_sg / container_sg** (`perceiving-objects`):
  `robot.get_observation` → the bundle's canonical DINO + VLM + SAM3
  script → `geometry.filter_and_compute_obb`. Outputs (`target_obb`,
  `container_obb`, …) bind to downstream subgraph inputs by name.
- **grasp_sg** (`grasping-direct-ik`): open →
  `geometry.top_down_grasp_candidates(obb)` → pre-rotate at altitude →
  straight-line descend (`robot.go_to_pose`, in-process IK — no planner) →
  close. Guarded by a `validate=True` checkpoint (`target_held`) evaluated
  against simulator ground truth.
- **transport_sg** (`transporting-objects`): drop pose from the container
  OBB → lift + lateral waypoint legs → descend, release, retract.

## The builder API

The grasp subgraph, as
[build_graph.py](gh-engine:examples/build_a_graph/build_graph.py)
actually writes it:

```python
from gap.builder import Ref, Subgraph, Workflow

sg = Subgraph(name="grasp_sg", skill="grasping-direct-ik")
sg.add_input("target_obb", type_name="OrientedBoundingBox")
sg.add_node("open", type="tool", tool="robot.open_gripper",
            inputs={"settle_steps": 40})
sg.add_node("compute_grasp", type="tool",
            tool="geometry.top_down_grasp_candidates",
            inputs={"obb": Ref("in.target_obb")})
sg.add_node(
    "compute_align", type="script", script="scripts/compute_align_pose.py",
    inputs={"grasp_pose": Ref("compute_grasp.candidates.poses.0"),
            "target_obb": Ref("in.target_obb")},
)
sg.add_node("rotate_align", type="tool", tool="robot.go_to_pose",
            inputs={"pose": Ref("compute_align.align_pose")})
sg.add_node("descend", type="tool", tool="robot.go_to_pose",
            inputs={"pose": Ref("compute_grasp.candidates.poses.0")})
sg.add_node("close", type="tool", tool="robot.close_gripper",
            inputs={"settle_steps": 60})
sg.add_exit("grasped")            # noop success marker
sg.set_on_error("failed")         # any raise → the "failed" exit
for src, dst in [("START", "open"), ("open", "compute_grasp"),
                 ("compute_grasp", "compute_align"),
                 ("compute_align", "rotate_align"),
                 ("rotate_align", "descend"), ("descend", "close"),
                 ("close", "grasped"), ("grasped", "END")]:
    sg.add_edge(src, dst)
```

The align-then-descend split avoids twisting the gripper against the
object while closing. `Ref("node.field.path")` is the dataflow primitive
(it serializes to the JSON `{"$ref": ...}` form); `Ref("in.<name>")` reads
a declared subgraph input. Cross-subgraph dataflow needs no explicit
wiring — a subgraph input named `target_obb` binds to whichever upstream
subgraph output has that name.

At the top level, the workflow routes each subgraph's exit value with
`add_conditional_edges(..., router_field="exit")`, and the failure end
node carries **recovery actions** the executor runs on the way out:

```python
wf.add_node("done", type="end", status="success")
wf.add_node(
    "abort", type="end", status="failure",
    recovery=[{"tool": "robot.open_gripper", "inputs": {}},
              {"tool": "robot.go_home", "inputs": {}}],
)
```

See [Builder](../authoring/builder.md) for the full API and the
[workflow schema](../reference/workflow-schema.md) for what it serializes
to.

## The checkpoint sidecar

`write_graph` emits `checkpoints/grasp_sg.py`, a postcondition the
executor evaluates against the sim's ground-truth world snapshot every
time `grasp_sg` exits on its success path:

```python
from gap.runtime.verify import Checkpoint


def _target_held(world) -> bool:
    """The target is in contact with a robot finger link after `close`."""
    return world.body('alphabet soup').is_grasped()


CHECKPOINTS = [
    Checkpoint(
        name="target_held",
        subgraph="grasp_sg",
        predicate=_target_held,
        rationale=(
            "After close, the grasp target must actually be held — "
            "ground-truth contacts prove the gripper closed ON the object, "
            "not on air."
        ),
        validate=True,
    ),
]
```

Checkpoints only evaluate in sim (they need the privileged
`world_snapshot`); `gap run` defaults to `--checkpoints warn`. See
[Checkpoints](../running/checkpoints.md).

:::{note}
Scripts are copied into `<out>/scripts/` **before** `wf.save(...)` —
`save` parses and structurally validates the workflow, importing each
script for schema introspection, so it must see the complete artifact.
After saving, `build_graph.py` re-validates with the skill registry in the
loop, the same checks as `gap run <dir> --validate-only`.
:::

## Execute it

```bash
MUJOCO_GL=egl uv run python examples/build_a_graph/build_graph.py --out my_graph --execute
# or, identically:
MUJOCO_GL=egl uv run gap run my_graph --sim libero_object_all_variance/0
uv run gap viz     # browse the recorded trace
```

With `--execute`, the script uses the Python execution API directly:

```python
import gap

conn = gap.connector.sim("libero", task="libero_object_all_variance/0")
try:
    result = gap.execute(out_dir, conn, skills=skills_root)
finally:
    conn.close()

result.success             # bool
result.exit_status         # which end node the run reached
result.duration_s          # wall-clock seconds
result.checkpoint_results  # per-checkpoint pass/fail
result.trace_path          # browse with `gap viz`
```

The open-robot-skills checkout is auto-discovered (`$GAP_SKILLS_PATH` or
the checkout next to the graph-as-policy checkout); pass
`--skills /path/to/open-robot-skills` to override.

## Where to go next

- [generate_a_graph](generate-a-graph.md) — let the LLM pipeline author
  this same artifact from a one-line instruction.
- [libero_quickstart](libero-quickstart.md) — the hand-tuned checked-in
  variant of this graph, with measured success rates.
- [Workflow schema](../reference/workflow-schema.md) — the JSON schema the
  builder targets; [Executor](../reference/executor.md) — its runtime
  semantics.
