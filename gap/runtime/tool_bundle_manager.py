"""Lifecycle manager for out-of-process tool bundle servers.

Parallel to :mod:`gap.runtime.policy_manager` — the policy manager owns
long-running learned-policy websocket servers; this one owns long-running
``stdio-msgpack`` tool servers (one subprocess per bundle that declares
``gap.serving.protocol: stdio-msgpack`` in its SKILL.md). The launcher
calls :meth:`boot_all` after policies are up; on workflow exit the caller
MUST invoke :meth:`shutdown_all` from a ``finally``.

A booted bundle's tools register with the gap-runtime ToolRegistry via
:meth:`gap_core.tools.ToolRegistry.register_rpc` so the executor sees the
RPC-routed tools side-by-side with the in-process @tool catalog. The LLM
catalog and the workflow validator are unaware of the transport — they
just see another ``runtime`` tool.

**One server per bundle per process, however many managers ask.** A batched
simulator runs W worlds in one process, each world its own thread, its own
graph, its own connector and its own manager — so twelve worlds booted twelve
of every bundle. Measured on one wave: 12 sam3 + 12 geometry + 12
grounding-dino, 6.3 GiB of GPU and 7.8 GiB of host RAM each for the segmenter
alone, ~20 s of model build apiece, and the whole set loaded while the
simulation sat frozen at its barrier — 138 s of a 205 s wave. It ended the way
that arithmetic has to end: ``Warp CUDA error 2: out of memory`` in the middle
of the first twelve-world iteration of a study run.

So a client is cached per process and handed to every manager that asks for
that bundle, with a reference count deciding when the subprocess dies. It is
safe by construction rather than by hope: ``ToolClient.call`` holds the
client's own lock across the whole request-and-reply, and the wire is strictly
sequential (one request, one reply, ids checked), so concurrent callers queue
instead of interleaving. What they buy for that queueing is measured too --
twelve worlds' perception serialised through one server cost ~1.2 s more per
episode than twelve parallel servers, against ~70 GiB of GPU recovered.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_SHARED_LOCK = threading.RLock()
"""Guards :data:`_SHARED`. Re-entrant because a boot failure releases under it."""

_SHARED: dict[tuple, "_SharedClient"] = {}
"""Live bundle clients, keyed by what makes two requests the same server.

Process-global on purpose: the whole point is that managers which know
nothing about each other -- one per world of a wave -- end up on one
subprocess. Nothing here is keyed by manager, and a manager that never shares
(the single-world case) sees a count of one and the old behaviour exactly.
"""


@dataclass
class _SharedClient:
    client: Any
    refs: int


def _client_key(name: str, bundle_dir: Any, env: Any) -> tuple:
    """What has to match for two managers to get the same server.

    The environment is part of it, not a detail: two graphs asking for the
    same bundle with different ``serving.env`` want different servers, and
    handing the second one the first's process would be a silent, wrong
    answer -- the failure mode this whole module exists to avoid.
    """
    if isinstance(env, dict):
        stamp = tuple(sorted((str(k), str(v)) for k, v in env.items()))
    else:
        stamp = (repr(env),)
    return (str(name), str(bundle_dir), stamp)


def _acquire_client(key: tuple, make: Any) -> tuple[Any, bool]:
    """The cached client for ``key``, or one built by ``make``. Returns
    ``(client, booted_now)``.

    ``make`` runs while the cache lock is held, which is deliberate: eleven
    threads waiting ~20 s for one model to load is the same wall clock as
    eleven threads each loading their own, and it is the only ordering where
    the second caller cannot start a second subprocess.
    """
    with _SHARED_LOCK:
        shared = _SHARED.get(key)
        if shared is not None:
            shared.refs += 1
            return shared.client, False
        client = make()
        _SHARED[key] = _SharedClient(client=client, refs=1)
        return client, True


def _release_client(key: tuple) -> None:
    """Give back one reference; close the subprocess when the last one goes."""
    with _SHARED_LOCK:
        shared = _SHARED.get(key)
        if shared is None:
            return
        shared.refs -= 1
        if shared.refs > 0:
            return
        _SHARED.pop(key, None)
    try:
        shared.client.close()
    except BaseException:
        logger.warning("closing a shared tool bundle client failed", exc_info=True)


class ToolBundleStartupError(RuntimeError):
    """A bundle's tool server failed to come up (or its catalog handshake
    failed). Surfaces the bundle name + the underlying error."""


@dataclass
class _Managed:
    bundle: str
    client: Any  # ToolClient
    tool_names: list[str]
    key: tuple  # into _SHARED; this manager holds one reference to it


class ToolBundleManager:
    """Boot every RPC-protocol tool bundle the workflow references; shut down on exit.

    Typical use (from the launcher)::

        manager = ToolBundleManager(skill_registry=skill_registry,
                                    tool_registry=tool_registry)
        manager.boot_all(["sam3", "curobo", "geometry"])  # noqa
        try:
            executor.execute()
        finally:
            manager.shutdown_all()
    """

    def __init__(self, skill_registry: Any, tool_registry: Any,
                 *, startup_timeout_s: float = 60.0,
                 evict_grace_s: float = 5.0,
                 call_timeout_s: float | None = None) -> None:
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._timeout = float(startup_timeout_s)
        self._grace = float(evict_grace_s)
        # Per-call reply timeout passed to each ToolClient. None lets the
        # client resolve GAP_TOOL_CALL_TIMEOUT_S / its built-in default.
        self._call_timeout_s = call_timeout_s
        self._lock = threading.Lock()
        self._managed: dict[str, _Managed] = {}

    def boot_all(self, bundle_names: Iterable[str]) -> None:
        """Spawn one server per bundle, register every catalog entry.

        Bundles whose SKILL.md does NOT declare ``serving.protocol ==
        "stdio-msgpack"`` are silently skipped — they run in-process.
        """
        from gap_core.rpc.client import ToolClient

        names = sorted(set(bundle_names))
        if not names:
            return
        started: list[_Managed] = []
        try:
            for name in names:
                info = self._skill_registry.get(name)
                serving = info.meta.serving
                if serving is None or serving.protocol != "stdio-msgpack":
                    continue
                key = _client_key(name, info.meta.bundle_dir, serving.env)
                try:
                    client, booted = _acquire_client(
                        key,
                        lambda name=name, info=info, serving=serving: ToolClient(
                            bundle_name=name,
                            bundle_dir=info.meta.bundle_dir,
                            env=serving.env,
                            evict_grace_s=self._grace,
                            call_timeout_s=self._call_timeout_s,
                        ),
                    )
                except BaseException as exc:
                    raise ToolBundleStartupError(
                        f"tool bundle {name!r}: failed to start its server "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc
                logger.info(
                    "[tool-bundle:%s] %s", name,
                    "booting" if booted else "reusing this process's server",
                )

                tool_names: list[str] = []
                for entry in client.catalog:
                    if entry.name in self._tool_registry:
                        existing = self._tool_registry.get(entry.name)
                        if existing.transport != "rpc":
                            # An in-process registration (a connector's own
                            # tool) claims this name: in-process wins, and the
                            # bundle's copy is simply not offered. A simulator
                            # connector providing geometry.* directly is the
                            # normal case, not an error.
                            logger.info(
                                "[tool-bundle:%s] %r already provided "
                                "in-process; keeping the existing registration",
                                name, entry.name,
                            )
                            continue
                    try:
                        self._tool_registry.register_rpc(
                            entry.name,
                            client,
                            summary=entry.summary,
                            tags=tuple(entry.tags),
                        )
                        tool_names.append(entry.name)
                    except ValueError as exc:
                        # Name collision between two RPC bundles exporting the
                        # same name. Surface as a startup error — silent
                        # dispatch would be confusing. Release rather than
                        # close: another world may be mid-call on this server.
                        _release_client(key)
                        raise ToolBundleStartupError(
                            f"tool bundle {name!r}: cannot register "
                            f"{entry.name!r} — {exc}"
                        ) from exc
                started.append(_Managed(bundle=name, client=client,
                                         tool_names=tool_names, key=key))
                logger.info("[tool-bundle:%s] %d tools registered",
                            name, len(tool_names))

            with self._lock:
                for m in started:
                    self._managed[m.bundle] = m
        except BaseException:
            # Give back what we took before re-raising. A reference, not a
            # close: a sibling world may be holding the same server.
            for m in started:
                _release_client(m.key)
            raise

    def shutdown_all(self) -> None:
        """Terminate every tool bundle subprocess we started. Idempotent."""
        with self._lock:
            managed = list(self._managed.values())
            self._managed.clear()

        if not managed:
            return
        logger.info("ToolBundleManager: releasing %d tool bundle%s",
                    len(managed), "" if len(managed) == 1 else "s")
        for m in managed:
            _release_client(m.key)

    def loaded_bundles(self) -> list[str]:
        return sorted(self._managed.keys())


def shared_client_counts() -> dict[str, int]:
    """``{bundle: managers holding it}`` for every live shared server.

    For tests and for anyone asking the question this module was changed to
    answer: how many servers is this process actually running?
    """
    with _SHARED_LOCK:
        return {key[0]: shared.refs for key, shared in _SHARED.items()}


__all__ = ["ToolBundleManager", "ToolBundleStartupError", "shared_client_counts"]
