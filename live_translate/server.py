"""Local web UI: FastAPI + WebSocket around the Pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import traceback
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from . import setup as model_setup
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
        # latest model-setup progress; None once every model is in place
        self.setup: dict | None = None
        self.setup_thread: threading.Thread | None = None

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
            "setup": self.setup,
            "muted": bool(p and p.muted),
        }

    # -- control -----------------------------------------------------------
    def start(self, s: dict) -> None:
        if (self.thread and self.thread.is_alive()) or self.setup:
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
            output_device=resolve_device(s.get("output_device"), "output"),
        )
        p.on_status = self._on_status
        p.on_update = lambda u: self.emit({"type": "utterance", "utterance": u.to_dict()})
        p.on_partial = lambda t: self.emit({"type": "partial", "text": t})
        p.on_provisional = lambda t: self.emit({"type": "provisional", "text": t})
        p.provisional = bool(s.get("provisional", True))
        p.polish = bool(s.get("polish", True))
        p.on_block = lambda b: self.emit({"type": "block", "block": b.to_dict()})
        p.on_level = lambda v: self.emit({"type": "level", "level": v})
        self.pipeline = p
        self.thread = threading.Thread(target=self._run, args=(p, s), daemon=True)
        self.thread.start()

    # -- model setup -----------------------------------------------------
    def run_setup(self) -> None:
        if self.setup_thread and self.setup_thread.is_alive():
            return
        self.setup = model_setup.Progress().to_dict()
        self.emit({"type": "setup", "setup": self.setup})
        self.setup_thread = threading.Thread(target=self._setup, daemon=True)
        self.setup_thread.start()

    def _setup(self) -> None:
        log = {"label": None, "tenth": -1}

        def report(p: model_setup.Progress) -> None:
            self.setup = None if p.phase == "done" else p.to_dict()
            self.emit({"type": "setup", "setup": self.setup})
            # a few plain lines for the terminal window -- the page has the real progress
            if p.phase == "download" and p.label != log["label"]:
                log["label"] = p.label
                print(f"  Lade {p.label} …", flush=True)
            if p.phase == "download" and p.total and p.done * 10 // p.total > log["tenth"]:
                log["tenth"] = p.done * 10 // p.total
                print(f"  {log['tenth'] * 10} %  ({model_setup.gb(p.done)} von {model_setup.gb(p.total)})", flush=True)
            if p.phase == "convert":
                print("  Albanisch-Modell wird umgewandelt …", flush=True)
            if p.phase == "done":
                print("  Modelle bereit.", flush=True)

        try:
            model_setup.run(report)
        except BaseException as e:
            traceback.print_exc()
            self.setup = model_setup.Progress("error", error=str(e) or type(e).__name__).to_dict()
            self.emit({"type": "setup", "setup": self.setup})

    def _on_status(self, status: str) -> None:
        self.status = status
        self.emit({"type": "status", "status": status})

    def _run(self, p: Pipeline, s: dict) -> None:
        try:
            p.load()
            if s.get("file"):
                p.run_file(s["file"], realtime=True, stop=self.stop)
            else:
                p.run_mic(resolve_device(s.get("input_device"), "input"), self.stop)
            p.wait_idle(timeout=60)
        # BaseException, not Exception: libraries in the model stack call sys.exit()
        # on setup problems, and threading swallows SystemExit without a word -- the
        # UI would sit on "Lade Modelle …" forever instead of showing what broke.
        except BaseException as e:
            traceback.print_exc()
            self._on_status(f"error: {type(e).__name__}: {e}".rstrip(": "))
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


def resolve_device(value: object, kind: str) -> int | None:
    """Device index for a name or an index. Names win, because PortAudio indices shift as
    soon as a device appears or disappears: the index the page stored yesterday may well be
    a different device today. Returns None (= system default) if nothing matches."""
    if value is None or value == "":
        return None
    import sounddevice as sd

    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    devs = sd.query_devices()
    if isinstance(value, str) and not value.lstrip("-").isdigit():
        for i, d in enumerate(devs):
            if d["name"] == value and d[channels] > 0:
                return i
        return None
    idx = int(value)
    return idx if 0 <= idx < len(devs) and devs[idx][channels] > 0 else None


session = Session()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/devices")
async def devices() -> JSONResponse:
    import sounddevice as sd

    # PortAudio reads the device list once, when it initializes, and never again -- so a
    # headset plugged in after the server started is invisible no matter how often the page
    # is reloaded. Re-initializing rebuilds the list, but it invalidates every open stream,
    # so only do it while no session is running.
    if not (session.thread and session.thread.is_alive()):
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            traceback.print_exc()

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
            elif cmd == "mute":
                if session.pipeline:
                    session.pipeline.muted = bool(msg.get("muted"))
                    session.emit({"type": "muted", "muted": session.pipeline.muted})
            elif cmd == "setup":
                if session.setup and session.setup["phase"] == "error":
                    session.run_setup()
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
    # missing models are fetched right away, with the progress on the page -- not on the
    # first "Start", where the UI would sit on "Lade Übersetzer …" for half an hour
    if model_setup.todo():
        print("  Sprachmodelle fehlen und werden jetzt geladen (Fortschritt im Browser).", flush=True)
        session.run_setup()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
