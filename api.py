"""
api.py

SANA synthetic image request helpers.
Only networking / retry / prompt selection lives here.
env.py calls this to fetch one synthetic grayscale image.
"""

import io
import json
import time
from typing import List, Tuple, Optional

import numpy as np
import requests
from PIL import Image

# Default SANA endpoint and prompts
DEFAULT_SANA_URL = "http://192.168.31.155:8000/generate"
DEFAULT_PROMPTS: List[str] = [
    "Congo red stain of amyloid deposition, 20x. Salmon-pink extracellular deposits in vessel walls and stroma, realistic microscopy lighting.",
    "H&E histopathology of colon adenocarcinoma, 40x. Irregular infiltrative glands, dirty necrosis, stromal desmoplasia, clinical photomicrograph.",
    "PAS stain histopathology of kidney glomerulus, 40x. Bright magenta thickened basement membranes, sharp counterstain, minimal background.",
]

# Default local LLM (vLLM OpenAI-compatible) endpoint and prompt schema
DEFAULT_LLM_URL = "http://192.168.31.155:8001/v1/chat/completions"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B"

DEFAULT_LLM_USER_INSTRUCTION = (
    "You generate ONLY ONE English histopathology prompt for diffusion models. "
    "Return a JSON object with a single key `prompt` (string). "
    "Constraints: 25–60 words, natural language only, one sentence with semicolons in this order: "
    "Organ + diagnosis, stain, standard magnification (4x/10x/20x/40x/60x or 100x oil); "
    "key microscopic features; irregular/non-symmetric tissue geometry (oblique cuts, fragmented fields, ragged contours) "
    "plus realistic artifacts; brightfield lighting/color calibration. "
    "No lists, no extra text."
)

DEFAULT_LLM_MESSAGES = [
    {"role": "user", "content": DEFAULT_LLM_USER_INSTRUCTION},
]

# vLLM guided JSON schema (structured output)
DEFAULT_LLM_GUIDED_JSON_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
    "additionalProperties": False,
}


def request_llm_histopath_prompt(
    llm_api_url: str = DEFAULT_LLM_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    messages: Optional[List[dict]] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 512,
    timeout: int = 120,
    retry_wait: int = 3,
    max_retries: int = 3,
    guided_json_schema: Optional[dict] = None,
    enable_thinking: Optional[bool] = None,
    extra_body: Optional[dict] = None,
) -> str:
    """Request ONE histopathology diffusion prompt from local LLM (OpenAI-compatible).

    The vLLM OpenAI server accepts guided decoding / structured-output parameters
    (e.g., guided_json) at top level, and other advanced options via extra_body.

    Returns:
        prompt: the generated prompt string.

    Raises:
        RuntimeError if all retries fail.
    """
    if messages is None:
        messages = DEFAULT_LLM_MESSAGES

    if guided_json_schema is None:
        guided_json_schema = DEFAULT_LLM_GUIDED_JSON_SCHEMA

    last_err = None
    for attempt in range(max_retries):
        try:
            payload = {
                "model": llm_model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
            }

            # vLLM structured output
            if guided_json_schema:
                payload["guided_json"] = guided_json_schema

            # vLLM / Qwen3 extra options
            if extra_body:
                payload["extra_body"] = dict(extra_body)

            if enable_thinking is not None:
                payload.setdefault("extra_body", {})
                ctk = payload["extra_body"].get("chat_template_kwargs", {})
                ctk["enable_thinking"] = enable_thinking
                payload["extra_body"]["chat_template_kwargs"] = ctk

            resp = requests.post(llm_api_url, json=payload, timeout=timeout)

            if resp.status_code == 429:
                print(
                    f"[LLM warn] LLM busy (429). sleep {retry_wait}s then retry... ({attempt + 1}/{max_retries})"
                )
                time.sleep(retry_wait)
                continue

            resp.raise_for_status()

            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            prompt = None
            # If guided_json was used, content should be JSON.
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    prompt = obj.get("prompt")
            except Exception:
                prompt = None

            if not prompt:
                prompt = content.splitlines()[0].strip() if content else ""

            if not prompt:
                raise ValueError("Empty prompt from LLM")
            
            print(f"[LLM info] obtained prompt: {prompt}")

            return prompt

        except Exception as e:
            last_err = e
            print(f"[LLM warn] prompt request failed: {e}. retry after {retry_wait}s")
            time.sleep(retry_wait)

    raise RuntimeError(
        f"LLM prompt request failed after {max_retries} retries: {last_err}"
    )


def request_sana_sample(
    api_url: str = DEFAULT_SANA_URL,
    prompts: Optional[List[str]] = None,  # legacy, no longer used for sampling
    height: int = 256,
    width: int = 256,
    guidance_scale: float = 4.5,
    num_inference_steps: int = 20,
    timeout: int = 180,
    retry_wait: int = 5,
    max_retries: int = 5,
    # --- LLM (Qwen3-8B via vLLM) parameters ---
    llm_api_url: str = DEFAULT_LLM_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_messages: Optional[List[dict]] = None,
    llm_temperature: float = 0.6,
    llm_top_p: float = 0.95,
    llm_top_k: int = 20,
    llm_max_tokens: int = 512,
    llm_timeout: int = 120,
    llm_retry_wait: int = 3,
    llm_max_retries: int = 3,
    llm_guided_json_schema: Optional[dict] = None,
    llm_enable_thinking: Optional[bool] = None,
    llm_extra_body: Optional[dict] = None,
) -> Tuple[np.ndarray, str]:
    """Request one synthetic grayscale image from SANA using a prompt from local LLM.

    Flow:
      1) Call local Qwen3-8B (vLLM OpenAI-compatible) to generate ONE prompt.
      2) Use that prompt to request ONE image from SANA.

    Returns:
        gt_np: uint8 numpy array shaped (height, width)
        prompt: the LLM-generated prompt used

    Raises:
        RuntimeError if LLM or SANA retries are exhausted.
    """
    # Step 1: get ONE prompt from LLM (no random sampling here).
    llm_prompt = request_llm_histopath_prompt(
        llm_api_url=llm_api_url,
        llm_model=llm_model,
        messages=llm_messages,
        temperature=llm_temperature,
        top_p=llm_top_p,
        top_k=llm_top_k,
        max_tokens=llm_max_tokens,
        timeout=llm_timeout,
        retry_wait=llm_retry_wait,
        max_retries=llm_max_retries,
        guided_json_schema=llm_guided_json_schema,
        enable_thinking=llm_enable_thinking,
        extra_body=llm_extra_body,
    )

    # Step 2: use ONLY this prompt to generate image from SANA.
    last_err = None
    for attempt in range(max_retries):
        try:
            payload = {
                "prompt": llm_prompt,
                "num_images": 1,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
            resp = requests.post(api_url, json=payload, timeout=timeout)

            if resp.status_code == 429:
                print(
                    f"[API warn] SANA busy (429). sleep {retry_wait}s then retry... ({attempt + 1}/{max_retries})"
                )
                time.sleep(retry_wait)
                continue

            resp.raise_for_status()

            pil_img = Image.open(io.BytesIO(resp.content)).convert("L")

            # Pillow 10+ uses Image.Resampling, older uses Image.BILINEAR
            resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            if pil_img.size != (width, height):
                pil_img = pil_img.resize((width, height), resample)

            gt_np = np.array(pil_img, dtype=np.uint8)
            return gt_np, llm_prompt

        except Exception as e:
            last_err = e
            print(f"[API warn] SANA request failed: {e}. retry after {retry_wait}s")
            time.sleep(retry_wait)

    raise RuntimeError(
        f"SANA synthetic request failed after {max_retries} retries: {last_err}"
    )