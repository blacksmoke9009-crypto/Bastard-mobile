import os
import secrets
from typing import List, Literal
import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="BASTARD Backend", version="0.1.0")

ACCESS_TOKEN = os.getenv("BASTARD_ACCESS_TOKEN", "")
PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY", "")
PROVIDER_BASE_URL = os.getenv("PROVIDER_BASE_URL", "https://api.openai.com/v1").rstrip("/")
PROVIDER_MODEL = os.getenv("PROVIDER_MODEL", "gpt-4o-mini")

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/health")
async def health():
    return {"status": "ok", "service": "bastard-backend"}

@app.post("/chat")
async def chat(request: ChatRequest, authorization: str | None = Header(default=None)):
    if ACCESS_TOKEN:
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied, ACCESS_TOKEN):
            raise HTTPException(status_code=401, detail="Token de acceso incorrecto.")
    if not PROVIDER_API_KEY:
        raise HTTPException(status_code=503, detail="Falta configurar PROVIDER_API_KEY en el servidor.")
    if not request.messages:
        raise HTTPException(status_code=400, detail="No hay mensajes para enviar.")

    payload = {
        "model": PROVIDER_MODEL,
        "messages": [m.model_dump() for m in request.messages],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {PROVIDER_API_KEY}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(f"{PROVIDER_BASE_URL}/chat/completions", json=payload, headers=headers)
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"El proveedor de IA respondió HTTP {response.status_code}.")
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        if isinstance(reply, list):
            reply = "".join(part.get("text", "") for part in reply if isinstance(part, dict))
        return {"reply": str(reply)}
    except HTTPException:
        raise
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        raise HTTPException(status_code=502, detail="No se pudo completar la solicitud al proveedor de IA.")
