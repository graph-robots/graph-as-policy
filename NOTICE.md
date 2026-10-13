# Third-party notices

gap and open-robot-skills are released under the MIT License (see
[LICENSE](LICENSE)). They build on the third-party projects below. Vendored
submodules keep their own LICENSE files in `third_party/`; pip/git
dependencies are installed from their upstream repositories and are never
vendored into this tree. **Model weights carry their own licenses,
independent of this repository's MIT license** — review them before
commercial use.

## Vendored submodules (`third_party/`)

| Project | Where | License (verified at the pinned revision) |
|---|---|---|
| [Variational-Automation-Benchmark](https://github.com/ehehee/Variational-Automation-Benchmark) (LIBERO fork with baked pose/permutation variations) | `third_party/Variational-Automation-Benchmark` | MIT — © 2023 Lifelong Robot Learning (LIBERO) |
| [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) (classic-suite assets/init states) | `third_party/LIBERO-PRO` | MIT — © 2023 Lifelong Robot Learning |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | `third_party/robosuite` | MIT — © 2022 Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab |
| robots_realtime (realtime Franka/gripper control loops; spawned as a subprocess, never imported) | `third_party/robots_realtime` | No license file at the pinned revision — used as an unmodified submodule pointer; consult the upstream repository before redistributing its code |

## Git/pip dependencies (installed by the user, never vendored)

| Project | Pulled in by | License |
|---|---|---|
| [pyroki](https://github.com/chungmin99/pyroki) (in-process IK, JAX) | gap core | MIT |
| [SAM3](https://github.com/facebookresearch/sam3) — model code via git pin; weights `facebook/sam3` from HuggingFace | the `sam3` bundle (`gap skills install sam3`) | **SAM License** (Meta custom license; governs both code and weights — review its use restrictions and redistribution terms) |
| Grounding DINO — used through `transformers`; weights `IDEA-Research/grounding-dino-base` downloaded from HuggingFace on first call | the `grounding-dino` bundle (`gap skills install grounding-dino`) | Apache-2.0 (GroundingDINO project and model card) |
| [NVIDIA cuRobo](https://github.com/NVlabs/curobo) (collision-aware motion planning) | the `curobo` bundle (`gap skills install curobo`) — **optional**, required only for the planner variant and the acceptance benchmark | The pinned cuRoboV2 research-release revision ships an Apache-2.0 LICENSE (+ a LICENSE_ASSETS file for bundled robot assets); other cuRobo revisions are distributed under the **NVIDIA Source Code License** (non-commercial terms). cuRobo is never vendored here — verify the LICENSE of the revision you install. |
| [openpi-client](https://github.com/Physical-Intelligence/openpi) (policy-server websocket client) | `gap[policy]`, `open-robot-skills[pi05-libero]` / `[molmoact-libero]` | Apache-2.0 |

## Model weights

`gap skills check --download` prefetches weights from HuggingFace
(`facebook/sam3`, `IDEA-Research/grounding-dino-base`); gated repos need
`HF_TOKEN`. Policy presets (`gap policy serve pi05-libero`) download
checkpoints published by the openpi project. Weights are cached locally and
are never redistributed by this repository.

## Provenance

Parts of this codebase were ported from a private research codebase by the
same authors (gRPC service implementations refactored into in-process tool
bundles, runtime and verification libraries de-protoized). Comments noting
"ported from the dev tree" refer to that lineage.
