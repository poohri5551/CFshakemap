from typing import Optional
from pathlib import Path
import time, threading

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ==== ดึงฟังก์ชันจาก logic.py ของโปรเจกคุณ ====
# - fetch_latest_event_in_thailand() : ดึงเมตาเหตุการณ์ล่าสุด
# - compute_overlay_from_event(ev)   : คำนวณ/สร้างผลลัพธ์แผนที่จาก ev
# - simulate_event(lat, lon, depth_km, mag) : จำลองเหตุการณ์
from .logic import (
    fetch_latest_event_in_thailand,
    compute_overlay_from_event,
    simulate_event,
)

app = FastAPI(title="SHAKEMAP API", version="1.2.0")

# CORS (เปิดกว้างสำหรับ dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://eqshakemap.pages.dev",
        "https://map.shakemap.org",
        "https://shakemap.org",          # <— เพิ่มบรรทัดนี้
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ================== In-memory Cache (คำนวณครั้งแรก) ==================
_CACHE_LOCK = threading.Lock()
_CACHE = {
    "data": None,        # JSON ผลลัพธ์เต็ม (รวม data URL/HTML/เมตา)
    "event_key": None,   # คีย์อ้างอิงเหตุการณ์ล่าสุดที่คำนวณแล้ว
    "ts": 0.0,           # เวลาที่คำนวณ (epoch)
}

# ตั้ง TTL ถ้าอยากให้รีเฟรชอัตโนมัติเมื่อพ้นเวลา; None = ไม่หมดอายุเอง
CACHE_TTL_SEC: Optional[int] = None  # เช่น 600 = 10 นาที

def _make_event_key(meta: dict) -> str:
    return f"{meta.get('time_utc') or meta.get('time_th')}|{meta.get('lat')}|{meta.get('lon')}|{meta.get('mag')}|{meta.get('depth_km')}"

def _get_cached_ok() -> bool:
    if _CACHE["data"] is None:
        return False
    if CACHE_TTL_SEC is None:
        return True
    return (time.time() - (_CACHE["ts"] or 0)) < CACHE_TTL_SEC

def _compute_and_store() -> dict:
    """คำนวณผลจากเหตุการณ์ล่าสุด แล้วเก็บลงแคช (ต้องเรียกภายใต้ LOCK)"""
    ev = fetch_latest_event_in_thailand()
    data = compute_overlay_from_event(ev)
    meta = data.get("meta", {})
    _CACHE["data"] = data
    _CACHE["event_key"] = _make_event_key(meta)
    _CACHE["ts"] = time.time()
    return data

def _get_or_compute(force: bool = False) -> dict:
    # ถ้ามีแคชและไม่ force -> คืนเลย
    if not force and _get_cached_ok():
        return _CACHE["data"]
    # กันชนกัน
    with _CACHE_LOCK:
        # เช็กซ้ำใน critical section กัน race
        if not force and _get_cached_ok():
            return _CACHE["data"]
        return _compute_and_store()


# ================== Routes ==================

@app.get("/")
def index():
    # เสิร์ฟหน้าเว็บ
    return FileResponse(str(STATIC_DIR / "index.html"))

# GET สำหรับเปิดในเบราว์เซอร์/เทส
@app.get("/api/run")
def api_run_get():
    try:
        data = _get_or_compute(force=False)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# POST ใช้งานจริงจากหน้าเว็บ
@app.post("/api/run")
def api_run(body: dict = Body(default_factory=dict)):
    """
    โหมดปกติ: POST /api/run  (body ว่างก็ได้)
    โหมดจำลอง: POST /api/run { "mode":"simulate", "lat":..., "lon":..., "depth":..., "mag":... }
    บังคับรีเฟรช: POST /api/run { "force": true }
    """
    try:
        # โหมดจำลอง
        if body.get("mode") == "simulate":
            lat   = float(body["lat"])
            lon   = float(body["lon"])
            depth = float(body["depth"])
            mag   = float(body["mag"])
            data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
            return JSONResponse(data)

        force = bool(body.get("force"))
        data = _get_or_compute(force=force)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# รีเฟรชเหตุการณ์ล่าสุดแบบบังคับ (สำหรับแอดมิน/DevTools)
@app.post("/api/refresh")
def api_refresh():
    try:
        data = _get_or_compute(force=True)
        return JSONResponse({"ok": True, "meta": data.get("meta", {}), "event_key": _CACHE["event_key"]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ดูสถานะแคช (debug)
@app.get("/api/cache_state")
def api_cache_state():
    return {
        "has_cache": _CACHE["data"] is not None,
        "event_key": _CACHE["event_key"],
        "ts": _CACHE["ts"],
        "ttl_sec": CACHE_TTL_SEC,
    }

@app.post("/api/simulate")
def api_simulate(body: dict = Body(...)):
    try:
        lat   = float(body["lat"])
        lon   = float(body["lon"])
        depth = float(body["depth"])
        mag   = float(body["mag"])
        data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)