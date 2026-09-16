"""Local web UI: FastAPI + WebSocket around the Pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .config import PROFILES, TRANSLATORS, TTS_VOICE, default_translator
from .pipeline import Pipeline, Utterance

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="live-translate")

VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]


class Session:
    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.status = "idle"
        self.settings: dict = {}
        self.last_transcript: Path | None = None

    # -- broadcasting from worker threads ---------------------------------
    def emit(self, msg: dict) -> None:
        if self.loop is None:
            return
        data = json.dumps(msg, ensure_ascii=False)
        self.loop.call_soon_threadsafe(asyncio.ensure_future, self._send_all(data))

    async def _send_all(self, data: str) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def snapshot(self) -> dict:
        p = self.pipeline
        return {
            "type": "snapshot",
            "status": self.status,
            "settings": self.settings,
            "utterances": [u.to_dict() for u in p.utterances] if p else [],
            "blocks": [b.to_dict() for b in p.blocks] if p else [],
            "options": {
                "langs": {k: v.name for k, v in PROFILES.items()},
                "translators": list(TRANSLATORS),
                "default_translator": default_translator(),
                "voices": VOICES,
            },
            "transcript_path": str(self.last_transcript) if self.last_transcript else None,
        }

    # -- control -----------------------------------------------------------
    def start(self, s: dict) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.settings = s
        self.stop.clear()
        self.last_transcript = None
        p = Pipeline(
            lang=PROFILES[s.get("lang", "de")],
            translator_name=s.get("translator", default_translator()),
            tts_enabled=bool(s.get("tts", True)),
            voice=s.get("voice", TTS_VOICE),
            context_turns=int(s.get("context", 3)),
            output_device=s.get("output_device"),
        )
        p.on_status = self._on_status
        p.on_update = lambda u: self.emit({"type": "utterance", "utterance": u.to_dict()})
        p.on_partial = lambda t: self.emit({"type": "partial", "text": t})
        p.on_provisional = lambda t: self.emit({"type": "provisional", "text": t})
        p.provisional = bool(s.get("provisional", True))
        p.polish = bool(s.get("polish", True))
        p.on_block = lambda b: self.emit({"type": "block", "block": b.to_dict()})
        self.pipeline = p
        self.thread = threading.Thread(target=self._run, args=(p, s), daemon=True)
        self.thread.start()

    def _on_status(self, status: str) -> None:
        self.status = status
        self.emit({"type": "status", "status": status})

    def _run(self, p: Pipeline, s: dict) -> None:
        try:
            p.load()
            if s.get("file"):
                p.run_file(s["file"], realtime=True, stop=self.stop)
            else:
                p.run_mic(s.get("input_device"), self.stop)
            p.wait_idle(timeout=60)
        except Exception as e:  # surface model/device errors in the UI
            self._on_status(f"error: {e}")
            return
        finally:
            try:
                p.close()
            except Exception:
                pass
        self.last_transcript = p.save_transcript()
        self._on_status("idle")
        self.emit({"type": "saved", "path": str(self.last_transcript) if self.last_transcript else None})

    def request_stop(self) -> None:
        self.stop.set()
        self._on_status("stopping")


session = Session()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/devices")
async def devices() -> JSONResponse:
    import sounddevice as sd

    devs = sd.query_devices()
    default_in, default_out = sd.default.device
    return JSONResponse(
        {
            "inputs": [{"id": i, "name": d["name"], "default": i == default_in} for i, d in enumerate(devs) if d["max_input_channels"] > 0],
            "outputs": [{"id": i, "name": d["name"], "default": i == default_out} for i, d in enumerate(devs) if d["max_output_channels"] > 0],
        }
    )


@app.get("/export")
async def export(format: str = "md") -> PlainTextResponse:
    p = session.pipeline
    if p is None:
        return PlainTextResponse("", status_code=404)
    if format == "json":
        return PlainTextResponse(json.dumps([u.to_dict() for u in p.utterances], ensure_ascii=False, indent=2), media_type="application/json")
    if format == "txt":
        return PlainTextResponse(p.transcript_text(), media_type="text/plain; charset=utf-8")
    if format == "en":
        return PlainTextResponse(p.transcript_text(english_only=True), media_type="text/plain; charset=utf-8")
    return PlainTextResponse(p.transcript_markdown(), media_type="text/markdown; charset=utf-8")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session.loop = asyncio.get_running_loop()
    session.clients.add(websocket)
    await websocket.send_text(json.dumps(session.snapshot(), ensure_ascii=False))
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start":
                session.start(msg.get("settings", {}))
            elif cmd == "stop":
                session.request_stop()
            elif cmd == "clear":
                if session.pipeline and not (session.thread and session.thread.is_alive()):
                    session.pipeline.utterances.clear()
                    session.pipeline.blocks.clear()
                    session.pipeline.history.clear()
                await websocket.send_text(json.dumps(session.snapshot(), ensure_ascii=False))
    except WebSocketDisconnect:
        session.clients.discard(websocket)


def main() -> None:
    ap = argparse.ArgumentParser(prog="live-translate-ui")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"live-translate UI: {url}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
