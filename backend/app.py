import logging
import os
import sys
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from backend.model_inference import VoiceVerifier
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from model_inference import VoiceVerifier

app = FastAPI(title="VoiceMatch AI Backend")
logging.basicConfig(level=logging.INFO)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled error in FastAPI request")
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {str(exc)}"},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    backend_root = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(backend_root)
    preferred_model_dir = os.path.join(backend_root, "models", "ecapa_librispeech_finetuned")
    fallback_model_dir = os.path.join(project_root, "backend", "models", "ecapa_librispeech_finetuned")
    def _resolve_model_dir() -> str:
        if os.path.isdir(preferred_model_dir) and os.path.exists(os.path.join(preferred_model_dir, "hyperparams.yaml")):
            return preferred_model_dir
        if os.path.isdir(fallback_model_dir) and os.path.exists(os.path.join(fallback_model_dir, "hyperparams.yaml")):
            return fallback_model_dir
        return preferred_model_dir

    model_dir_to_use = _resolve_model_dir()
    verifier = VoiceVerifier(model_dir=model_dir_to_use)
except Exception as exc:
    verifier = None
    startup_error = str(exc)
else:
    startup_error = None


@app.post("/api/verify")
async def verify_voices(
    voice1: UploadFile = File(...),
    voice2: UploadFile = File(...),
):
    temp_dir = tempfile.mkdtemp()
    
    try:
        if verifier is None:
            raise HTTPException(
                status_code=503,
                detail=f"Model backend is unavailable: {startup_error}",
            )

        path1 = os.path.join(temp_dir, f"voice1_{voice1.filename}")
        path2 = os.path.join(temp_dir, f"voice2_{voice2.filename}")

        logging.info(f"Receiving voice1: {voice1.filename} (size: {voice1.size} bytes)")
        logging.info(f"Receiving voice2: {voice2.filename} (size: {voice2.size} bytes)")

        with open(path1, "wb") as buffer:
            shutil.copyfileobj(voice1.file, buffer)

        with open(path2, "wb") as buffer:
            shutil.copyfileobj(voice2.file, buffer)

        size1 = os.path.getsize(path1)
        size2 = os.path.getsize(path2)
        logging.info(f"Files saved: path1={size1} bytes, path2={size2} bytes")

        result = verifier.compare_voices(path1, path2, threshold=0.75)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Inference request failed: {str(e)}")
        logging.error(f"Voice1: {voice1.filename}, Voice2: {voice2.filename}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
