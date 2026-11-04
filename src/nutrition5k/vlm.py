"""Zero-shot estimates from vision-language models (VLMs), as a reference for the trained models.

The VLMs see the same plate crop and are scored with the same metrics, but they are never trained on Nutrition5k.
Every model is reached through an OpenAI-compatible chat API: Gemini and DeepSeek in the cloud, and llama.cpp's
``llama-server`` for a local model (Qwen3-VL on a pod). Only the standard library is used, so no SDK is needed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .dataset import CROP_BOX, TARGET_NAMES, Dataset, split_validation
from .evaluation import evaluate_predictions, write_json, write_predictions
from .models.average import AverageBaseline

# Provider → (OpenAI-compatible base URL, environment variable with the API key).
PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "local": ("http://127.0.0.1:8080/v1", None),  # llama-server
}
# One zero-shot prompt for every model, without hints about the camera or Nutrition5k. Fixed before the test call.
PROMPT = (
    "This is a top-down photo of one dish of food on a plate. Estimate the totals for all the food on the plate: "
    "mass in grams, energy in kilocalories, and fat, carbohydrate and protein in grams. Answer with only a JSON "
    'object with numbers: {"mass_g": 0, "calories_kcal": 0, "fat_g": 0, "carbohydrate_g": 0, "protein_g": 0}'
)
# JSON key → target, in TARGET_NAMES order.
FIELDS = dict(zip(("mass_g", "calories_kcal", "fat_g", "carbohydrate_g", "protein_g"), TARGET_NAMES))


@dataclass(frozen=True)
class VLMConfig:
    model: str  # model id at the provider, e.g. "gemini-3.8-flash"
    provider: str  # key of PROVIDERS
    split: str = "test"  # "test", or "val" to check the prompt before the test call
    limit: int | None = None  # only the first N dishes
    image_size: int = 448  # side of the plate crop sent (448 = native, no resize)
    workers: int = 8  # parallel requests
    tries: int = 3  # per dish, for HTTP errors and unparseable answers

    def name(self, dataset_mode: str) -> str:
        limit = f"-n{self.limit}" if self.limit else ""
        return f"{self.model.replace('/', '_')}-{self.split}{limit}-{dataset_mode}"


def run_vlm(dataset: Dataset, output_root: Path, config: VLMConfig) -> Path:
    """Ask the VLM about every dish once, then write predictions and evaluation like a trained run.

    Answers go to ``responses.jsonl`` as they arrive, so an interrupted run resumes with the missing dishes. A
    dish whose answer still cannot be parsed after ``tries`` gets the training means (the average baseline) and
    is counted in ``parse_failures``.
    """
    base_url, key_variable = PROVIDERS[config.provider]
    key = os.environ.get(key_variable) if key_variable else None
    if key_variable and not key:
        raise RuntimeError(f"set {key_variable} to use {config.provider}")
    output = output_root / dataset.mode / config.name(dataset.mode)
    output.mkdir(parents=True, exist_ok=True)
    train, validation = split_validation(dataset.train, 0.1, dataset.groups)
    records = [record for record in (validation if config.split == "val" else dataset.test) if record.image]
    records = records[: config.limit] if config.limit else records
    write_json(output / "config.json", {
        **asdict(config), "base_url": base_url, "prompt": PROMPT, "crop_box": list(CROP_BOX),
        "dishes": len(records), "started_at": datetime.now().isoformat(timespec="seconds"),
    })

    log = output / "responses.jsonl"
    answers = {row["dish_id"]: row for row in _read_jsonl(log) if row["values"]}
    lock = threading.Lock()

    def ask(record):
        row = _ask_dish(base_url, key, config, record)
        with lock, log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return row

    missing = [record for record in records if record.dish_id not in answers]
    print(f"{output.name}: {len(records) - len(missing)} done, {len(missing)} to ask")
    with ThreadPoolExecutor(config.workers) as pool:
        try:
            for count, row in enumerate(pool.map(ask, missing), 1):
                answers[row["dish_id"]] = row
                if count % 50 == 0 or count == len(missing):
                    print(f"  {count}/{len(missing)}")
        except BaseException:
            pool.shutdown(cancel_futures=True)  # do not send the queued dishes after a fatal error
            raise

    means = AverageBaseline.fit(train).means
    predictions = {r.dish_id: tuple(answers[r.dish_id]["values"] or means) for r in records}
    suffix = "_val" if config.split == "val" else ""
    write_predictions(output / f"predictions{suffix}.csv", predictions)
    write_json(output / f"evaluation{suffix}.json", evaluate_predictions(tuple(records), predictions))
    rows = [answers[r.dish_id] for r in records]
    write_json(output / "summary.json", {
        "dishes": len(records),
        "parse_failures": sum(not row["values"] for row in rows),
        "prompt_tokens": sum(row["usage"].get("prompt_tokens", 0) for row in rows),
        "completion_tokens": sum(row["usage"].get("completion_tokens", 0) for row in rows),
        # Some providers leave the thinking tokens out of completion_tokens; total - prompt counts them.
        "total_tokens": sum(row["usage"].get("total_tokens", 0) for row in rows),
        "mean_seconds": sum(row["seconds"] for row in rows) / len(rows),  # per request, incl. network and queue
    })
    return output


def parse_answer(text: str) -> tuple[float, ...] | None:
    """The five targets from the last JSON object in ``text``, or None if one is missing or not a number ≥ 0."""
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text or "")):
        try:
            values = json.loads(candidate)
            numbers = tuple(float(values[field]) for field in FIELDS)
        except (ValueError, KeyError, TypeError):
            continue
        if all(number >= 0 for number in numbers):
            return numbers
    return None


def plate_jpeg(path: Path, size: int) -> str:
    """The plate crop (``CROP_BOX``) at ``size`` px, as base64 JPEG."""
    x0, y0, side = CROP_BOX
    with Image.open(path) as image:
        crop = image.convert("RGB").crop((x0, y0, x0 + side, y0 + side))
    if size != side:
        crop = crop.resize((size, size), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode()


def _ask_dish(base_url: str, key: str | None, config: VLMConfig, record) -> dict:
    """One dish: up to ``tries`` requests until the answer parses. Keeps the last answer and the summed usage."""
    image = plate_jpeg(record.image, config.image_size)
    usage, text, error, began = {}, None, None, time.perf_counter()
    for attempt in range(config.tries):
        try:
            text, call_usage = _chat(base_url, key, config.model, image)
            error = None
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404):  # wrong key, model or request: stop at the first dish
                raise RuntimeError(f"{exc.code} from {base_url}: {exc.read().decode(errors='replace')[:500]}")
            text, call_usage, error = None, {}, repr(exc)
            time.sleep(2 ** (attempt + 2))
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            text, call_usage, error = None, {}, repr(exc)
            time.sleep(2 ** (attempt + 2))  # 4, 8, 16 s: rate limits and transient server errors
        for name, value in call_usage.items():
            if isinstance(value, int):
                usage[name] = usage.get(name, 0) + value
        values = parse_answer(text)
        if values:
            break
    return {"dish_id": record.dish_id, "values": values, "text": text, "error": error, "tries": attempt + 1,
            "usage": usage, "seconds": time.perf_counter() - began}


def _chat(base_url: str, key: str | None, model: str, image: str, timeout=600) -> tuple[str, dict]:
    """One chat completion with the prompt and the image; the provider's default sampling and thinking."""
    body = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
    ]}]}
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    request = urllib.request.Request(f"{base_url}/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return data["choices"][0]["message"]["content"], data.get("usage") or {}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
