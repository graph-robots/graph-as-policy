# build_a_graph — author a workflow in Python with `gap.builder`

> **What:** The full authoring example: checkpoints, recovery, `--execute` · **Needs:** `uv sync` to build · **Time:** ~1 min

LLM generation is one producer of graphs, not the only one. This example
builds the complete quickstart-style pick-and-place workflow by hand with
the builder API — **the builder is the same artifact pipeline as
`gap generate`**: it emits the identical workflow directory
(`workflow.json` + `scripts/` + `checkpoints/`), passes through the same
parser and structural validation as agent output, and runs with the same
`gap run` / `gap.execute`.

```bash
uv run python examples/build_a_graph/build_graph.py --out my_graph   # build + validate
```

prints the validation report and leaves a runnable artifact:

```
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

## What gets built

The graph is real, not a toy — it mirrors the
[libero_quickstart](../libero_quickstart/) structure and runs on the same
task (`libero_object_all_variance/0`):

```
START → target → container → grasp → transport → done
           ↘ not_found       ↘ failed   ↘ blocked → abort (open gripper, go home)
```

- **target_sg / container_sg** (`perceiving-objects`): `robot.get_observation`
  → the bundle's canonical DINO + VLM + SAM3 script →
  `geometry.filter_and_compute_obb`. Outputs (`target_obb`, `container_obb`,
  …) bind to downstream subgraph inputs by name.
- **grasp_sg** (`grasping-direct-ik`): open →
  `geometry.top_down_grasp_candidates(obb)` → pre-rotate at altitude →
  straight-line descend (`robot.go_to_pose`, in-process IK — no planner) →
  close. Guarded by a `validate=True` checkpoint (`target_held`) evaluated
  against simulator ground truth.
- **transport_sg** (`transporting-objects`): drop pose from the container
  OBB → lift + lateral waypoint legs → descend, release, retract.

## The builder API in 20 lines

```python
from gap.builder import Workflow, Subgraph, Ref

sg = Subgraph(name="grasp_sg", skill="grasping-direct-ik")
sg.add_input("target_obb", type_name="OrientedBoundingBox")
sg.add_node("open", type="tool", tool="robot.open_gripper")
sg.add_node("compute_grasp", type="tool",
            tool="geometry.top_down_grasp_candidates",
            inputs={"obb": Ref("in.target_obb")})
sg.add_node("descend", type="tool", tool="robot.go_to_pose",
            inputs={"pose": Ref("compute_grasp.candidates.poses.0")})
sg.add_node("close", type="tool", tool="robot.close_gripper")
sg.add_exit("grasped")            # noop success marker
sg.set_on_error("failed")         # any raise → the "failed" exit
for src, dst in [("START", "open"), ("open", "compute_grasp"),
                 ("compute_grasp", "descend"), ("descend", "close"),
                 ("close", "grasped"), ("grasped", "END")]:
    sg.add_edge(src, dst)

wf = Workflow(name="pick_into_basket")
wf.add_subgraph(sg)               # + perception / transport subgraphs
wf.add_node("grasp", type="subgraph", ref="grasp_sg")
# ... conditional edges on each subgraph's exit value ...
wf.save("my_graph/workflow.json")   # parse + structural validation
```

`Ref("node.field.path")` is the dataflow primitive (it serializes to the
JSON `{"$ref": ...}` form); `Ref("in.<name>")` reads a declared subgraph
input. Cross-subgraph dataflow needs no explicit wiring — a subgraph input
named `target_obb` binds to whichever upstream subgraph output has that
name.

See [build_graph.py](build_graph.py) for the full program, including how
the canonical bundle scripts are copied in and how the checkpoint sidecar
is written.

## Execute it

Same requirements as the quickstart (`uv sync && uv run gap skills install sam3 grounding-dino geometry` with
downloaded weights + a VLM credential):

```bash
MUJOCO_GL=egl uv run python examples/build_a_graph/build_graph.py --out my_graph --execute
# or, identically:
MUJOCO_GL=egl uv run gap run my_graph --sim libero_object_all_variance/0
uv run gap viz     # browse the recorded trace
```

The open-robot-skills checkout is auto-discovered (`$GAP_SKILLS_PATH` or the
checkout next to the graph-as-policy checkout); pass `--skills /path/to/open-robot-skills` to
override.

## Where to go next

- [generate_a_graph](../generate_a_graph/) — let the LLM pipeline author
  this same artifact from a one-line instruction.
- [libero_quickstart](../libero_quickstart/) — the hand-tuned checked-in
  variant of this graph, run end-to-end against sim ground truth.
- [grocery_packing](../grocery_packing/) — the looped version: the same
  perceive → grasp → transport, repeated until every item is packed.
- `docs/runtime.md` — the JSON schema and executor semantics the builder
  targets.
