# server/app.py
import json
import queue
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="server/static"), name="static")


@dataclass
class LatestState:
    a: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    result: Optional[np.ndarray] = None
    shape: Optional[Tuple[int, int, int]] = None  # (M,K,N)


latest = LatestState()
event_q: "queue.Queue[str]" = queue.Queue(maxsize=10000)


def publish(evt: dict) -> None:
    evt["ts"] = time.time()
    try:
        event_q.put_nowait(json.dumps(evt))
    except queue.Full:
        # Drop events if UI can't keep up; MVP behavior.
        pass


@app.get("/", response_class=HTMLResponse)
def index():
    with open("server/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/state")
def state():
    # Small JSON for initialization; matrices can be large, so we cap.
    def pack(mat: Optional[np.ndarray]):
        if mat is None:
            return None
        m, n = mat.shape
        cap = 32
        view = mat[: min(m, cap), : min(n, cap)]
        return {"shape": [int(m), int(n)], "data": view.astype(np.float32).tolist()}

    return {
        "shape": list(latest.shape) if latest.shape else None,
        "a": pack(latest.a),
        "b": pack(latest.b),
        "result": pack(latest.result),
    }


@app.get("/events")
def events():
    def gen():
        # SSE stream
        while True:
            msg = event_q.get()
            yield f"data: {msg}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def parse_mm_request(body: bytes) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
    """
    Binary protocol:
    - 4 bytes: magic 'CMM1'
    - 3x int64: M,K,N little-endian
    - raw float32 A of size M*K
    - raw float32 B of size K*N
    """
    if len(body) < 4 + 8 * 3:
        raise ValueError("body too small")
    magic = body[:4]
    if magic != b"CMM1":
        raise ValueError("bad magic")
    M, K, N = struct.unpack_from("<qqq", body, 4)
    off = 4 + 8 * 3
    a_n = M * K
    b_n = K * N
    a_bytes = a_n * 4
    b_bytes = b_n * 4
    need = off + a_bytes + b_bytes
    if len(body) != need:
        raise ValueError(f"bad body size: got={len(body)} need={need}")

    a = np.frombuffer(body, dtype=np.float32, count=a_n, offset=off).reshape(M, K).copy()
    off2 = off + a_bytes
    b = np.frombuffer(body, dtype=np.float32, count=b_n, offset=off2).reshape(K, N).copy()
    return M, K, N, a, b

import asyncio
delay = 1.0

@app.post("/mm")
async def mm(req: Request):
    body = await req.body()
    M, K, N, a, b = parse_mm_request(body)

    # Checks (mirror C++ side, defensive)
    if a.dtype != np.float32 or b.dtype != np.float32:
        return Response(content=b"bad dtype", status_code=400)
    if a.ndim != 2 or b.ndim != 2:
        return Response(content=b"bad dim", status_code=400)
    if a.shape != (M, K) or b.shape != (K, N):
        return Response(content=b"bad shape", status_code=400)
    if (M % 4) or (K % 4) or (N % 4):
        return Response(content=b"shape not multiple of 4", status_code=400)

    latest.a = a
    latest.b = b
    latest.shape = (int(M), int(K), int(N))
    latest.result = None

    publish({"type": "input", "M": int(M), "K": int(K), "N": int(N)})
    await asyncio.sleep(delay)

    result = np.zeros((M, N), dtype=np.float32)
    latest.result = result

    # Blocked multiply with visualization events
    block = 4
    for i in range(0, M, block):
        for j in range(0, N, block):
            block_c = np.zeros((block, block), dtype=np.float32)
            publish({"type": "block_start", "i": i, "j": j})
            await asyncio.sleep(delay)

            for k in range(0, K, block):
                a_blk = a[i : i + block, k : k + block]
                b_blk = b[k : k + block, j : j + block]
                publish({"type": "block_mul", "i": i, "j": j, "k": k})
                await asyncio.sleep(delay)

                partial = (a_blk @ b_blk).astype(np.float32)
                publish({"type": "partial", "i": i, "j": j, "k": k, "partial": partial.tolist()})
                await asyncio.sleep(delay)

                block_c += partial
                publish({"type": "accum", "i": i, "j": j, "k": k, "block_c": block_c.tolist()})
                await asyncio.sleep(delay)

            result[i : i + block, j : j + block] = block_c
            publish({"type": "writeback", "i": i, "j": j, "block_c": block_c.tolist()})
            await asyncio.sleep(delay) 

    latest.result = result
    publish({"type": "done"})
    await asyncio.sleep(delay)

    # Return raw float32 result
    return Response(content=result.tobytes(order="C"), media_type="application/octet-stream")