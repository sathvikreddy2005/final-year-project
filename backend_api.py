import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent

def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _select_python(env_var: str, local_venv: Path | None = None) -> Path:
    override = os.getenv(env_var)
    if override:
        return Path(override)
    if local_venv:
        candidate = _venv_python(local_venv)
        if candidate.exists():
            return candidate
    return Path(sys.executable)


QML_PYTHON = _select_python("TEXT_PYTHON", ROOT / "qml_env")
AUDIO_PYTHON = _select_python("AUDIO_PYTHON", ROOT / "Audio_Mental_Health_Project" / "audio_env")
VIDEO_PYTHON = _select_python("VIDEO_PYTHON", ROOT / "mental_health_project" / "venv")

TEXT_SCRIPT = ROOT / "TEXT" / "text_input_inference.py"
AUDIO_SCRIPT = (
    ROOT
    / "Audio_Mental_Health_Project"
    / "Audio_Mental_Health_Project"
    / "predict_audio.py"
)
VIDEO_SCRIPT = ROOT / "mental_health_project" / "video_analyzer.py"
AUDIO_WORKDIR = AUDIO_SCRIPT.parent
VIDEO_WORKDIR = VIDEO_SCRIPT.parent
UI_DIR = ROOT / "UI"


app = FastAPI(title="MindScope API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


class TextRequest(BaseModel):
    text: str


def _extract_json(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON found in output: {stdout[:400]}")
    return json.loads(stdout[start : end + 1])


def _run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 180) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return _extract_json(proc.stdout)


def _score_dict(pred: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ("stress", "depression", "anxiety"):
        if key not in pred:
            continue
        value = pred.get(key, 0.0)
        if isinstance(value, dict):
            out[key] = float(value.get("score", 0.0))
        else:
            out[key] = float(value)
    return out


async def _save_upload_to_tempfile(file: UploadFile, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())
    return tmp_path


def _cleanup_tempfile(path: Path | None) -> None:
    try:
        if path and path.exists():
            path.unlink()
    except Exception:
        pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "qml_python_exists": QML_PYTHON.exists(),
        "audio_python_exists": AUDIO_PYTHON.exists(),
        "video_python_exists": VIDEO_PYTHON.exists(),
        "text_script_exists": TEXT_SCRIPT.exists(),
        "audio_script_exists": AUDIO_SCRIPT.exists(),
        "video_script_exists": VIDEO_SCRIPT.exists(),
    }


@app.get("/")
def root():
    return RedirectResponse(url="/ui/mental-health-detection.html")


@app.post("/predict/text")
def predict_text(req: TextRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text input is required.")
    if not QML_PYTHON.exists():
        raise HTTPException(status_code=500, detail=f"Missing interpreter: {QML_PYTHON}")

    try:
        raw = _run_cmd(
            [str(QML_PYTHON), str(TEXT_SCRIPT), "--text", req.text, "--json-only"],
            cwd=ROOT,
            timeout=300,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text prediction failed: {exc}") from exc

    return {"modality": "text", "raw": raw, "scores": _score_dict(raw)}


@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    if not AUDIO_PYTHON.exists():
        raise HTTPException(status_code=500, detail=f"Missing interpreter: {AUDIO_PYTHON}")

    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    if suffix.lower() not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    try:
        tmp_path = await _save_upload_to_tempfile(file, suffix)

        raw = _run_cmd(
            [
                str(AUDIO_PYTHON),
                str(AUDIO_SCRIPT),
                "--file",
                str(tmp_path),
                "--json-only",
            ],
            cwd=AUDIO_WORKDIR,
            timeout=120,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio prediction failed: {exc}") from exc
    finally:
        _cleanup_tempfile(locals().get("tmp_path"))

    return {"modality": "audio", "raw": raw, "scores": _score_dict(raw)}


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if not VIDEO_PYTHON.exists():
        raise HTTPException(status_code=500, detail=f"Missing interpreter: {VIDEO_PYTHON}")

    suffix = Path(file.filename or "clip.webm").suffix or ".webm"
    video_exts = {".webm", ".mp4", ".mov", ".avi", ".mkv"}
    normalized_suffix = suffix.lower()
    if normalized_suffix not in video_exts:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    try:
        tmp_path = await _save_upload_to_tempfile(file, suffix)

        raw = _run_cmd(
            [
                str(VIDEO_PYTHON),
                str(VIDEO_SCRIPT),
                "--video",
                str(tmp_path),
                "--json-only",
            ],
            cwd=VIDEO_WORKDIR,
            timeout=180,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video prediction failed: {exc}") from exc
    finally:
        _cleanup_tempfile(locals().get("tmp_path"))

    return {"modality": "video", "raw": raw, "scores": _score_dict(raw)}
