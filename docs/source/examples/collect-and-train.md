# Collect and Train

:::{note} Requirements
A CUDA GPU with `MUJOCO_GL=egl`, the `quickstart` and `policy` extras, and a
VLM credential (the collection graph's perception uses one). Training itself
happens outside GaP with your framework of choice.
:::

[examples/collect_and_train](gh-engine:examples/collect_and_train) closes the
data loop: a verified GaP graph acts as a **scripted expert**, its rollouts
become a demonstration dataset, an external recipe trains a VLA policy on
them, and the trained policy comes back into GaP as its own **policy-skill**
node (like the shipped `pi05-libero` / `molmoact-libero` skills) — steered by
the same perception that collected the data ([Steered Policy](steered-policy.md)).

The whole example is one script,
[collect.py](gh-engine:examples/collect_and_train/collect.py), plus the
engine's `gap.connector.collector.DataCollector`.

## 1. Collect demonstrations

```bash
uv sync --extra policy && uv run gap skills install sam3 grounding-dino geometry   # sim + perception + openpi client

uv run python examples/collect_and_train/collect.py --episodes 50 --out demos.hdf5 \
    --graph examples/libero_quickstart/graph
```

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | `25` | number of rollouts to record |
| `--out` | `demos.hdf5` | output HDF5 file |
| `--graph` | `examples/libero_quickstart/graph` | the expert graph to execute |
| `--skills` | auto-discovered | open-robot-skills checkout path |
| `--suite` | `libero_object_all_variance` | LIBERO suite |
| `--task` | `0` | task id within the suite |

The script is short enough to read in full — the core loop is:

```python
import gap
from gap.connector.collector import DataCollector

with gap.connector.sim("libero", task=f"{args.suite}/{args.task}") as conn:
    collector = DataCollector(conn, args.out)
    try:
        for episode in range(args.episodes):
            conn.reset(seed=episode + 1)   # seeds index the baked variations
            collector.start_episode()
            result = gap.execute(args.graph, conn, skills=args.skills,
                                 checkpoints="warn")
            success, reward = conn.check_success()
            collector.end_episode(success=success)
    finally:
        collector.close()
```

`DataCollector` hooks the connector's step-callback seam, so while an episode
is open it records one synchronized row per control step — no changes to the
graph or the executor. Each episode resets to a fresh **baked** variation
seed (deterministic initial states, not RNG), runs the expert graph under
[checkpoint warnings](../running/checkpoints.md), and tags the episode with
the simulator's own success verdict from `conn.check_success()`.

## 2. The HDF5 dataset

`DataCollector` writes a flat, append-only layout:

```text
/observations/<camera>_rgb   uint8   [N, H, W, 3]
/observations/state          float32 [N, dof+1]   (arm joints + gripper)
/actions                     float32 [N, A]       (zero-padded to max A)
/rewards                     float32 [N]
/dones                       bool    [N]
/episode_ends                int64   [E]   (exclusive end index per episode)
/episode_success             bool    [E]
```

This maps 1:1 onto a LeRobot dataset: `/observations/<camera>_rgb` becomes
`observation.images.<camera>`, `/observations/state` becomes
`observation.state`, `/actions` becomes `action`, and
`episode_index` / `frame_index` columns are derived by binning row indices
with `/episode_ends`.

## 3. Convert and train (external)

Convert with a small custom LeRobot adapter over the layout above, **filter
to `episode_success == True` episodes**, then train with your recipe of
choice — e.g. [LeRobot](https://github.com/huggingface/lerobot) ACT/diffusion
baselines, or [openpi](https://github.com/Physical-Intelligence/openpi)
fine-tuning (`pi05` configs). GaP takes no position on the trainer: it only
needs the result behind a websocket policy server.

:::{tip}
The collection script prints `kept` (successful episodes) as it runs.
Because the expert graph is verified and the simulator scores every episode,
filtering to successes is a one-liner at conversion time rather than a manual
review pass.
:::

## 4. Serve and run the trained policy

A shipped policy skill auto-boots its preset, so a steered graph referencing
`pi05-libero` / `molmoact-libero` needs nothing extra; to run one server by
hand:

```bash
uv run gap policy serve pi05-libero --port 9100
```

For your own checkpoint, the cleanest path is to package it as its own
policy-skill bundle (subclass `gap.runtime.policy_skill.PolicyLoopSkill` with
its own preset). For a quick one-off, reference it from the graph by a name and
override that name in the config's `policies:` block with a `start_cmd` that
carries a `{port}` placeholder:

```yaml
policies:
  my_policy:
    start_cmd: "python serve_my_policy.py --checkpoint ckpt/ --port {port}"
```

Then run it inside a steered graph:

```bash
MUJOCO_GL=egl uv run gap run examples/steered_policy/graph_loop \
    --sim libero_object_all_variance/0
```

The steered graph hovers above the perceived object and hands control to your
policy — and the hand-off pose matches the distribution this collection
script produced, because both use the same OBB perception and approach.

## Next steps

- [Steered Policy](steered-policy.md) — the hybrid graphs that run the
  trained policy, and their hand-off tuning.
- [Policies](../benchmarks/policies.md) — serving presets, custom servers,
  and the policy registry.
- [Benchmark Grids](benchmark.md) — score the trained policy as the
  `policy_only` / `llm_plus_policy` benchmark modes.
