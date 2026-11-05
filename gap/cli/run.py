"""``gap run`` subcommand — execute (or validate) a workflow graph."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "run",
        help="Execute a gap workflow graph",
    )
    sp.add_argument(
        "graph",
        help="Path to a workflow directory or workflow.json",
    )
    sp.add_argument(
        "--sim", default=None, metavar="SUITE/TASK",
        help="Run against a sim connector, e.g. libero_object/0 "
             "(default: tools-only, no robot)",
    )
    sp.add_argument(
        "--real", default=None, choices=["franka", "ur_zed", "ur5e_bimanual"],
        help="Run against a real-hardware connector (franka: robots_realtime "
             "msgpack bridge; ur_zed: perception-only UR + ZED; ur5e_bimanual: "
             "the dionysus rig — GAP_UR_DRY_RUN=1 / GAP_UR_CONFIRM=1 recommended). "
             "READ THE SAFETY NOTES in examples/real_franka_pick_place/README.md "
             "before driving hardware.",
    )
    sp.add_argument(
        "--rr-config", default=None, metavar="YAML",
        help="franka only: rr-session config (relative to "
             "third_party/robots_realtime); default "
             "configs/franka/franka_robotiq_client.yaml",
    )
    sp.add_argument(
        "--no-rr-autostart", action="store_true",
        help="franka only: do not spawn rr-session; run it yourself in a "
             "second terminal",
    )
    sp.add_argument(
        "--skills", action="append", default=None, metavar="PATH",
        help="Skill registry root(s); repeatable, precedence-ordered. "
             "Default: the resolved registry set — $GAP_SKILLS_PATH, "
             "project [tool.gap], user config, or an open-robot-skills "
             "checkout next to the graph-as-policy checkout "
             "(see `gap registry list`)",
    )
    sp.add_argument(
        "--validate-only", action="store_true",
        help="Validate the workflow graph without executing it",
    )
    sp.add_argument(
        "--no-trace", action="store_true",
        help="Disable trace output (default: traces into ./outputs/run_<timestamp>)",
    )
    sp.add_argument(
        "--trace-dir", default=None,
        help="Trace output directory (overrides the default outputs/run_<timestamp>)",
    )
    sp.add_argument(
        "--no-video", action="store_true",
        help="Sim only: disable run-video recording. Video is ON by default "
             "for sim runs (saved to <trace-dir>/run_video.mp4, plus per-camera "
             "videos when the env buffers them); pass this to skip it (faster, "
             "no rendering).",
    )
    sp.add_argument(
        # Deprecated: video now records by default for sim runs. Kept as an
        # accepted no-op so existing scripts/examples don't break.
        "--record-video", action="store_true", help=argparse.SUPPRESS,
    )
    sp.add_argument(
        "--checkpoints", default="warn", choices=["off", "warn", "raise"],
        help="Checkpoint enforcement mode (default: warn)",
    )
    sp.add_argument(
        "--inputs", nargs="*", default=[], metavar="K=V",
        help="Initial workflow inputs as k=v pairs (values parsed as JSON "
             "when possible, else strings)",
    )
    sp.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sp.set_defaults(func=_handle)


def _parse_inputs(pairs: list[str]) -> dict:
    import json

    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--inputs entries must be k=v, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except (ValueError, TypeError):
            out[key] = raw
    return out


def _validate_only(graph: str, skills: list[str] | None) -> int:
    from pathlib import Path

    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow

    path = Path(graph)
    if path.is_dir():
        path = path / "workflow.json"
    try:
        wf = load_workflow(path)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    from gap.skills import load_registry_set, resolve_registries

    registry_set = resolve_registries(skills)
    skill_registry = load_registry_set(registry_set) if registry_set else None

    issues = validate_workflow(wf, skill_registry=skill_registry)
    for issue in issues:
        print(str(issue))
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        print(f"FAIL: {len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
        return 1
    print(f"OK: 0 errors, {len(issues)} warning(s)")
    return 0


def _handle(args: argparse.Namespace) -> int:
    import logging
    import time
    from pathlib import Path

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.validate_only:
        return _validate_only(args.graph, args.skills)

    # Tracing is ON by default: ./outputs/run_<timestamp> unless overridden.
    if args.no_trace:
        trace_dir = None
    elif args.trace_dir:
        trace_dir = args.trace_dir
    else:
        # Claim the directory atomically: the timestamp alone collides when
        # two runs launch in the same second (observed — their node_data
        # and videos interleaved silently).
        base = Path("outputs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        candidate, n = base, 1
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                n += 1
                candidate = base.with_name(f"{base.name}_{n}")
        trace_dir = str(candidate)

    if args.sim and args.real:
        raise SystemExit("--sim and --real are mutually exclusive")

    # Video records by default for sim runs. It needs both a sim connector
    # (to render) and a trace dir (to save into), so it's silently skipped
    # for real/tools-only runs or when tracing is off. `--no-video` opts out.
    record_video = bool(args.sim) and trace_dir is not None and not args.no_video
    if args.sim and args.no_trace and not args.no_video:
        print("note: skipping run video — needs a trace dir (drop --no-trace)")

    connector = None
    if args.sim:
        import gap.connector

        connector = gap.connector.sim(
            "libero", task=args.sim, record_video=record_video,
        )
        if record_video:
            # Capture only arms itself on reset(); execute() runs on the
            # already-reset env, so start the frame buffer explicitly.
            connector.start_video()
    elif args.real:
        import gap.connector

        connector = gap.connector.real(
            args.real,
            rr_config=args.rr_config,
            rr_autostart=not args.no_rr_autostart,
        )

    from gap.runtime.execute import execute

    success_metric: tuple[bool, float] | None = None
    try:
        result = execute(
            args.graph,
            connector,
            skills=args.skills,
            inputs=_parse_inputs(args.inputs) or None,
            trace_dir=trace_dir,
            checkpoints=args.checkpoints,
        )
        save_video = getattr(connector, "save_video", None)
        if record_video and save_video is not None and trace_dir is not None:
            video_path = Path(trace_dir) / "run_video.mp4"
            saved = save_video(str(video_path))
            if saved.get("success") and saved.get("num_frames"):
                print(f"video: {video_path} ({saved['num_frames']} frames)")
            else:
                print(f"video: not saved ({saved})")
        # Snapshot success/reward BEFORE close() — the sim env tears
        # down its MjSim there and compute_reward()/task_completed()
        # would crash on a closed env.
        check_success = getattr(connector, "check_success", None) if connector else None
        if check_success is not None:
            try:
                success_metric = check_success()
            except Exception:
                success_metric = None
    finally:
        if connector is not None:
            connector.close()

    status = "SUCCESS" if result.success else "FAILURE"
    print(f"{status} (exit={result.exit_status}, {result.duration_s:.1f}s)")
    if success_metric is not None:
        completed, reward = success_metric
        print(f"reward: {reward:.4f}  success={bool(completed)}")
    if result.latency:
        steps = int(result.latency.get("control_steps", 0))
        freq = float(result.latency.get("control_freq", 0.0))
        sim_phys = float(result.latency.get("sim_physics_wall_s", 0.0))
        ctrl_s = steps / freq if freq > 0 else 0.0
        compute_s = max(0.0, result.duration_s - sim_phys)
        physical_s = ctrl_s + compute_s
        print(f"sim wallclock: {result.duration_s:.2f} s")
        print(
            f"physical execution: {physical_s:.2f} s  "
            f"(control: {ctrl_s:.2f} s @ {freq:.1f} Hz × {steps} steps; "
            f"compute: {compute_s:.2f} s)"
        )
    if result.trace_path is not None:
        print(f"trace: {result.trace_path}")
    for cp in result.checkpoint_results:
        print(f"checkpoint: {cp}")
    if result.error is not None:
        print(f"error: {result.error}")
    return 0 if result.success else 1
