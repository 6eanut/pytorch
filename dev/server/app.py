import json
import queue
import struct
import time
import asyncio
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
    shape: Optional[Tuple[int, int, int]] = None

latest = LatestState()
event_q: "queue.Queue[str]" = queue.Queue(maxsize=10000)

def publish(evt: dict) -> None:
    evt["ts"] = time.time()
    try:
        event_q.put_nowait(json.dumps(evt))
    except queue.Full:
        pass

@app.get("/", response_class=HTMLResponse)
def index():
    with open("server/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/state")
def state():
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
        while True:
            msg = event_q.get()
            yield f"data: {msg}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

def parse_mm_request(body: bytes) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
    if len(body) < 28:
        raise ValueError("body too small")
    magic = body[:4]
    if magic != b"CMM1":
        raise ValueError("bad magic")
    M, K, N = struct.unpack_from("<qqq", body, 4)
    off = 28
    a_n, b_n = M * K, K * N
    a_bytes, b_bytes = a_n * 4, b_n * 4
    a = np.frombuffer(body, dtype=np.float32, count=a_n, offset=off).reshape(M, K).copy()
    b = np.frombuffer(body, dtype=np.float32, count=b_n, offset=off + a_bytes).reshape(K, N).copy()
    return M, K, N, a, b

delay = 1.0 # Faster for visualization

@app.post("/mm")
async def mm(req: Request):
    body = await req.body()
    M, K, N, a, b = parse_mm_request(body)
    
    latest.a, latest.b = a, b
    latest.shape = (int(M), int(K), int(N))
    latest.result = np.zeros((M, N), dtype=np.float32)
    
    publish({"type": "input", "M": int(M), "K": int(K), "N": int(N)})
    await asyncio.sleep(delay*2)
    
    block = 4
    for i in range(0, M, block):
        for j in range(0, N, block):
            block_c = np.zeros((block, block), dtype=np.float32)
            
            for k in range(0, K, block):
                # 1. Split
                publish({"type": "split", "i": i, "j": j, "k": k})
                await asyncio.sleep(delay)
                
                # 2. Feed to Helper
                a_blk = a[i : i + block, k : k + block]
                b_blk = b[k : k + block, j : j + block]
                publish({"type": "feed", "i": i, "j": j, "k": k})
                await asyncio.sleep(delay)
                
                # 3. Store Temporary
                partial = (a_blk @ b_blk).astype(np.float32)
                publish({"type": "storage", "i": i, "j": j, "k": k})
                await asyncio.sleep(delay)
                
                # 4. Accumulate
                block_c += partial
                publish({"type": "accumulate", "i": i, "j": j, "k": k})
                await asyncio.sleep(delay)
            
            # 5. Writeback
            latest.result[i : i + block, j : j + block] = block_c
            publish({"type": "writeback", "i": i, "j": j})
            await asyncio.sleep(delay)
            
    publish({"type": "done"})
    return Response(content=latest.result.tobytes(order="C"), media_type="application/octet-stream")