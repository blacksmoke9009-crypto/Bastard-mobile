
import os
import secrets
from typing import List, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


app = FastAPI(title="BASTARD Backend", version="0.2.0")

ACCESS_TOKEN = os.getenv("BASTARD_ACCESS_TOKEN", "")
PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY", "")
PROVIDER_BASE_URL = os.getenv(
    "PROVIDER_BASE_URL",
    "https://api.openai.com/v1"
).rstrip("/")

PROVIDER_MODEL = os.getenv("PROVIDER_MODEL", "gpt-4o-mini")
IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "bytedance-seed/seedream-4.5"
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class ImageRequest(BaseModel):
    prompt: str
    image_base64: str | None = None
    mime_type: str = "image/jpeg"


def check_access(authorization: str | None):
    if ACCESS_TOKEN:
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not secrets.compare_digest(supplied, ACCESS_TOKEN):
            raise HTTPException(
                status_code=401,
                detail="Token de acceso incorrecto."
            )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "bastard-backend"
    }


@app.post("/chat")
async def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None)
):
    check_access(authorization)

    if not PROVIDER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Falta configurar PROVIDER_API_KEY en el servidor."
        )

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="No hay mensajes para enviar."
        )

    payload = {
        "model": PROVIDER_MODEL,
        "messages": [m.model_dump() for m in request.messages],
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {PROVIDER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{PROVIDER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers
            )

        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"El proveedor de IA respondió HTTP {response.status_code}."
            )

        data = response.json()
        reply = data["choices"][0]["message"]["content"]

        if isinstance(reply, list):
            reply = "".join(
                part.get("text", "")
                for part in reply
                if isinstance(part, dict)
            )

        return {"reply": str(reply)}

    except HTTPException:
        raise
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="No se pudo completar la solicitud al proveedor de IA."
        )


@app.post("/images")
async def generate_image(
    request: ImageRequest,
    authorization: str | None = Header(default=None)
):
    check_access(authorization)

    if not PROVIDER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Falta configurar PROVIDER_API_KEY en el servidor."
        )

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Escribí una descripción para generar la imagen."
        )

    payload = {
        "model": IMAGE_MODEL,
        "prompt": request.prompt.strip(),
        "n": 1
    }

    if request.image_base64:
        mime = request.mime_type.strip().lower()

        if mime not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(
                status_code=400,
                detail="Formato de imagen no compatible."
            )

        image_data = request.image_base64

        if "," in image_data and image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]

        payload["input_references"] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{image_data}"
                }
            }
        ]

    headers = {
        "Authorization": f"Bearer {PROVIDER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{PROVIDER_BASE_URL}/images",
                json=payload,
                headers=headers
            )

        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"El proveedor de imágenes respondió HTTP {response.status_code}."
            )

        data = response.json()
        image = data["data"][0]["b64_json"]
        mime_type = data["data"][0].get("media_type", "image/png")

        return {
            "image_base64": image,
            "mime_type": mime_type,
            "cost": data.get("usage", {}).get("cost")
        }

    except HTTPException:
        raise
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="No se pudo generar la imagen."
        )
