"""Msgpack-over-TCP bridge for robots_realtime.

Implements the framing protocol the real-robot stack uses to talk to the
robots_realtime client (vendored at ``third_party/robots_realtime``; see
its ``robots_realtime/utils/server_client_utils.py`` for the client side).

Protocol:
  - robots_realtime connects as a TCP **client** to this server
  - Each exchange: client sends a framed observation, server replies with
    the latest action
  - Frame format: 4-byte big-endian length header + msgpack payload
  - Numpy arrays are supported via msgpack_numpy

Wire formats:

  Observation (client → server):
    {
        b'left': {b'joint_pos': np.float32(8,)},
        b'camera_top': {
            b'images': {b'left_rgb': np.uint8(H,W,3)},
            b'depth_data': np.float32(H,W),
            b'intrinsics': {b'left': {b'intrinsics_matrix': np.float32(3,3)}},
            b'pose': np.float32(7,),
            b'pose_mat': np.float32(4,4),
        },
    }

  Action (server → client):
    {
        "timestamp": float,
        "left": {"joint_pos": [float]*7, "gripper": float},
    }
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
from typing import Any

import msgpack
import msgpack_numpy as m

m.patch()  # enable numpy array serialization

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Framing helpers
# ---------------------------------------------------------------------------

def encode_msg(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def decode_msg(raw: bytes) -> dict:
    return msgpack.unpackb(raw, raw=True)


async def send_framed(writer: asyncio.StreamWriter, obj: dict) -> None:
    payload = encode_msg(obj)
    header = struct.pack("!I", len(payload))
    writer.write(header + payload)
    await writer.drain()


async def recv_framed(reader: asyncio.StreamReader) -> dict:
    header = await reader.readexactly(4)
    (msg_len,) = struct.unpack("!I", header)
    payload = await reader.readexactly(msg_len)
    return decode_msg(payload)


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------

class MsgpackNumpyServer:
    """TCP server that exchanges observations and actions with robots_realtime."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port

        # Shared state — read/written by FrankaRealEnv
        self.latest_observation: dict[str, Any] | None = None
        self.latest_action: dict[str, Any] = {}
        #: Optional ``fn(observation)`` called for *every* received frame, on
        #: the server thread, before the reply. For envs that must not miss
        #: fields a client sends only occasionally (polling
        #: ``latest_observation`` sees just the newest frame).
        self.on_observation = None

        # Set by start() / start_server_in_background() for clean shutdown.
        self._asyncio_server: asyncio.AbstractServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        import socket
        server = await asyncio.start_server(
            self._handle, self.host, self.port,
            reuse_address=True,
            start_serving=False,
        )
        # Enable SO_REUSEADDR on all sockets to avoid "address already in use"
        for sock in server.sockets:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        await server.start_serving()
        self._asyncio_server = server
        logger.info("[MsgpackServer] listening on %s:%s", self.host, self.port)
        async with server:
            await server.serve_forever()

    def stop(self) -> None:
        """Stop serving from any thread (idempotent, best-effort).

        Closes the listening sockets and stops the event loop the server
        runs on; the background thread started by
        :func:`start_server_in_background` then exits.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return

        def _shutdown() -> None:
            if self._asyncio_server is not None:
                self._asyncio_server.close()
            loop.stop()

        loop.call_soon_threadsafe(_shutdown)

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        logger.info("[MsgpackServer] client connected")
        try:
            while True:
                # Receive observation from robots_realtime
                request = await recv_framed(reader)
                self.latest_observation = request
                hook = self.on_observation
                if hook is not None:
                    try:
                        hook(request)
                    except Exception:
                        logger.exception("[MsgpackServer] on_observation hook failed")

                # Reply with latest action
                await send_framed(writer, self.latest_action)
        except asyncio.IncompleteReadError:
            logger.info("[MsgpackServer] client disconnected")
        except Exception:
            logger.exception("[MsgpackServer] error")


def start_server_in_background(
    server: MsgpackNumpyServer,
) -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Start the msgpack server in a background daemon thread."""
    loop = asyncio.new_event_loop()
    server._loop = loop

    def _run() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.start())
        except RuntimeError:
            # loop.stop() during shutdown interrupts run_until_complete.
            pass
        finally:
            # Drain still-pending tasks (open client handlers, the
            # serve_forever task) so loop.close() doesn't warn about
            # destroyed-but-pending tasks.
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return loop, thread
