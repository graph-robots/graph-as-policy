---
name: Bug report
about: Something broke — a crash, a wrong result, a graph that won't validate or run
title: ""
labels: bug
assignees: ""
---

## What happened

A clear description of the bug, and what you expected instead.

## Reproduction

The exact command(s) or minimal Python snippet:

```bash
MUJOCO_GL=egl gap run ... --sim ... --skills ...
```

- Graph: (path to the example, or attach your `workflow.json`)
- Does `gap run <graph> --validate-only` pass?

## Trace

If the run produced a trace (`outputs/run_<timestamp>/`), attach
`dag_trace.json` and the failing node's `node_data/<id>/` folder — they
contain the resolved inputs/outputs and make most bugs reproducible
without your hardware.

## Environment

- OS / kernel:
- Python version:
- GPU + driver (`nvidia-smi` header) and `MUJOCO_GL` value, if sim-related:
- gap install: commit SHA + extras installed (e.g. `[libero]`)
- open-robot-skills: commit SHA + bundles installed (e.g. `sam3`, `curobo`)
- LLM/VLM provider + model (if agent/VLM-related):

## Logs

```
paste the relevant log tail here (run with -v for debug logging)
```

> Real-robot safety issues (a guard that can be bypassed, unexpected
> motion): mention it explicitly so it gets the `safety` label and
> priority review.
