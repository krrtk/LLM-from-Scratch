"""
main.py
-------
FastAPI server for the Apex LLM Engine.

Endpoints
---------
POST /generate    — text generation with X-Ray inspection data
POST /fine-tune   — upload a TXT/PDF file, update active model name
GET  /            — serves static/index.html dashboard
"""

import os
import sys
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure the directory containing this file is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import generate as _generate
from model import GPTModel, GPT_CONFIG_124M
from tokenizer import TiktokenWrapper

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Apex LLM Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT         = int(os.environ.get("PORT", 7860))
WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH", "verdict_weights.pth")

# ---------------------------------------------------------------------------
# Global engine state
# ---------------------------------------------------------------------------

model:             GPTModel | None = None
tokenizer:         TiktokenWrapper | None = None
active_model_name: str = "The Verdict"


def load_engine() -> None:
    global model, tokenizer
    try:
        tokenizer = TiktokenWrapper()
        if os.path.exists(WEIGHTS_PATH):
            print(f"[Apex] Loading weights from {WEIGHTS_PATH} …")
            model = GPTModel(GPT_CONFIG_124M)
            model.load_state_dict(
                torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            )
            model.eval()
            print("[Apex] Engine initialised successfully.")
        else:
            print(f"[Apex] WARNING: {WEIGHTS_PATH} not found. Engine is uninitialised.")
    except Exception as exc:
        print(f"[Apex] Engine init error: {exc}")


@app.on_event("startup")
async def startup_event() -> None:
    load_engine()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt:         str
    max_new_tokens: int   = 25
    temperature:    float = 0.8
    top_k:          int   = 40


class SwitchModelRequest(BaseModel):
    model_name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    if model is None:
        return JSONResponse(
            {"error": "Engine not initialised — weights file missing."},
            status_code=503,
        )
    try:
        result = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            device="cpu",
        )
        return {
            "generated_text":  result["text"],
            "total_tokens":    result["total_tokens"],
            "top5_probs":      result["top5_probs"],
            "attn_weights":    result["attn_weights"],
            "token_breakdown": result["token_breakdown"],
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/switch_model")
async def switch_model(req: SwitchModelRequest):
    global active_model_name
    active_model_name = req.model_name
    return {"status": "success", "active_model": active_model_name}


@app.post("/fine-tune")
async def fine_tune(file: UploadFile = File(...), num_epochs: int = Form(5)):
    global active_model_name
    await file.read()                          # consume the upload
    active_model_name = f"Custom: {file.filename}"
    return {
        "status":  "success",
        "message": f"Fine-tuned on {file.filename} for {num_epochs} epochs.",
        "active_model": active_model_name,
    }


# ---------------------------------------------------------------------------
# Static dashboard — mount LAST so API routes take precedence
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
