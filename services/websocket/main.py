"""
GraminRoute — WebSocket Server
===============================
Maintains persistent connections to:
- shop:{shop_id}  — each retailer's browser
- hub:{district}  — hub manager's dashboard

Other services push events via internal HTTP API:
  POST /internal/emit/shop/{shop_id}  {event, data}
  POST /internal/emit/hub/{district}  {event, data}
  POST /internal/emit/broadcast       {event, data}
"""

import os
import logging

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("websocket")

# FIX: use AsyncRedisManager so Socket.io can scale horizontally
# Falls back to in-memory if Redis isn't available
try:
    mgr = socketio.AsyncRedisManager(REDIS_URL)
    sio = socketio.AsyncServer(
        async_mode="asgi",
        client_manager=mgr,
        cors_allowed_origins="*",
        logger=False,
        engineio_logger=False,
    )
    log.info("✅ Socket.io using Redis manager")
except Exception as e:
    log.warning(f"Redis manager unavailable ({e}), using in-memory")
    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins="*",
        logger=False,
        engineio_logger=False,
    )

fastapi_app = FastAPI(title="GraminRoute WebSocket Server", version="1.0")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app = socketio.ASGIApp(sio, fastapi_app)


# ── Socket.io Event Handlers ────────────────

@sio.event
async def connect(sid, environ, auth):
    log.info(f"🔌 Client connected: {sid}")


@sio.event
async def disconnect(sid):
    log.info(f"🔌 Client disconnected: {sid}")


@sio.event
async def join_room(sid, data):
    """
    Client sends: emit('join_room', {room: 'shop:R001'})
    or:           emit('join_room', {room: 'hub:Jangaon'})
    """
    room = data.get("room", "")
    if room:
        # FIX: enter_room is synchronous in python-socketio — no await
        sio.enter_room(sid, room)
        log.info(f"📍 {sid} joined room: {room}")
        await sio.emit("room_joined", {"room": room}, to=sid)


@sio.event
async def leave_room(sid, data):
    room = data.get("room", "")
    if room:
        # FIX: leave_room is also synchronous
        sio.leave_room(sid, room)
        log.info(f"📍 {sid} left room: {room}")


# ── Internal HTTP API ────────────────────────

class EmitPayload(BaseModel):
    event: str
    data: Dict[str, Any]


@fastapi_app.post("/internal/emit/shop/{shop_id}")
async def emit_to_shop(shop_id: str, payload: EmitPayload):
    room = f"shop:{shop_id}"
    await sio.emit(payload.event, payload.data, room=room)
    log.info(f"📤 Emitted '{payload.event}' to {room}")
    return {"status": "sent", "room": room, "event": payload.event}


@fastapi_app.post("/internal/emit/hub/{district}")
async def emit_to_hub(district: str, payload: EmitPayload):
    room = f"hub:{district}"
    await sio.emit(payload.event, payload.data, room=room)
    log.info(f"📤 Emitted '{payload.event}' to {room}")
    return {"status": "sent", "room": room, "event": payload.event}


@fastapi_app.post("/internal/emit/broadcast")
async def emit_broadcast(payload: EmitPayload):
    await sio.emit(payload.event, payload.data)
    log.info(f"📤 Broadcast '{payload.event}' to all clients")
    return {"status": "broadcast", "event": payload.event}


@fastapi_app.get("/internal/rooms")
async def list_rooms():
    rooms = {}
    ns = sio.manager.rooms.get("/", {})
    for room_name in ns:
        participants = list(ns[room_name])
        rooms[room_name] = {"count": len(participants), "sids": participants[:10]}
    return rooms


@fastapi_app.get("/health")
async def health():
    return {"status": "ok", "service": "websocket"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
