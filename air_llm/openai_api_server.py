"""OpenAI-compatible HTTP API server for a single AirLLM model.

Exposes the standard OpenAI surface (POST /v1/chat/completions with optional
SSE streaming, GET /v1/models) on top of an AirLLM streaming model.

AirLLM streams weights layer-by-layer (or expert-by-expert) from disk on every
forward, so a single process can only run one generation at a time. All requests
are serialized through a lock; the generation itself runs on worker threads so
the asyncio event loop keeps serving health checks and can pump the SSE queue.

Usage:
    .conda_env/bin/python air_llm/openai_api_server.py \
        --model Qwen/Qwen3-30B-A3B-Instruct \
        --shards-path /home/kg/airllm_shards \
        --compression 4bit
"""
import argparse
import asyncio
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from transformers import TextIteratorStreamer

from airllm import AutoModel

# ---- CLI-provided configuration (set in main()) --------------------------------------------
MODEL_ID = None
SHARDS_PATH = None
COMPRESSION = None
HF_TOKEN = None
MAX_SEQ_LEN = 4096
MAX_ROUNDS = 4
DEFAULT_MAX_NEW_TOKENS = 256
NO_PREFETCHING = False

# ---- runtime model state, filled by lifespan -------------------------------------------------
model = None
tokenizer = None
gen_lock = threading.Lock()
_QUEUE_TIMEOUT = 600.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    user: Optional[str] = None
    max_rounds: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"[server] loading {MODEL_ID} ...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        layer_shards_saving_path=SHARDS_PATH,
        max_seq_len=MAX_SEQ_LEN,
        compression=COMPRESSION,
        prefetching=not NO_PREFETCHING,
        hf_token=HF_TOKEN,
    )
    tokenizer = model.tokenizer
    print(f"[server] loaded {MODEL_ID} in {time.time() - t0:.1f}s "
          f"(max_seq_len={model.max_seq_len})")
    yield
    model = None
    tokenizer = None


app = FastAPI(title="AirLLM OpenAI-compatible server", lifespan=lifespan)


def _window_messages(req: ChatCompletionRequest):
    """Keep the system prompt plus the most recent rounds of the conversation.

    ``truncation=True`` in the tokenizer drops the *tail* of the prompt, which for a
    chat means the newest turns get cut. Instead, window by rounds first so long
    conversations keep what the model actually needs to answer, then let the tokenizer
    truncation remain as a final safety net for a single over-long turn.

    A round is one (user, assistant) pair; a final pending user message is kept too.
    """
    msgs = [m.model_dump() for m in req.messages]

    max_rounds = req.max_rounds if req.max_rounds is not None else MAX_ROUNDS
    if max_rounds <= 0 or len(msgs) <= 2:
        return msgs

    head = 0
    while head < len(msgs) and msgs[head]["role"] == "system":
        head += 1
    system_msgs = msgs[:head]
    history = msgs[head:]

    pending = []
    if history and history[-1]["role"] != "assistant":
        pending = [history[-1]]
        history = history[:-1]

    if len(history) > max_rounds * 2:
        history = history[len(history) - max_rounds * 2:]

    return system_msgs + history + pending


def _prepare_inputs(req: ChatCompletionRequest):
    msgs = _window_messages(req)
    try:
        prompt = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    except Exception:
        prompt = "\n\n".join(f"{m.role}: {m.content}" for m in req.messages)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False,
                       truncation=True, max_length=MAX_SEQ_LEN, padding=False)
    input_ids = inputs["input_ids"].to(model.running_device)
    return input_ids, input_ids.shape[1]


def _build_generation_kwargs(req: ChatCompletionRequest, prompt_len: int):
    max_new = req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    budget = MAX_SEQ_LEN - prompt_len
    if budget <= 0:
        raise HTTPException(400, f"prompt longer than max_seq_len={MAX_SEQ_LEN}")
    max_new = min(max_new, budget)
    if max_new <= 0:
        raise HTTPException(400, "max_tokens must be > 0")

    gen_kwargs = dict(max_new_tokens=max_new, use_cache=True)

    if req.temperature is not None and req.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = req.temperature
    else:
        gen_kwargs["do_sample"] = False
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p
    if req.top_k is not None:
        gen_kwargs["top_k"] = req.top_k
    if req.presence_penalty:
        gen_kwargs["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty:
        gen_kwargs["frequency_penalty"] = req.frequency_penalty
    if req.seed is not None:
        gen_kwargs["seed"] = req.seed
    if req.n and req.n > 1:
        gen_kwargs["num_return_sequences"] = req.n

    stop_strings = []
    if req.stop:
        stop_strings += [req.stop] if isinstance(req.stop, str) else req.stop
    cfg_stop = getattr(model.generation_config, "stop_strings", None)
    if cfg_stop:
        stop_strings += list(cfg_stop)
    if stop_strings:
        gen_kwargs["stop_strings"] = stop_strings

    return gen_kwargs


def _sse(payload) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
async def root():
    return {"service": "AirLLM OpenAI-compatible server",
            "model": MODEL_ID,
            "endpoints": ["GET /v1/models", "POST /v1/chat/completions"]}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "airllm"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None:
        raise HTTPException(503, "model not loaded")
    if req.stream and req.n != 1:
        raise HTTPException(400, "n > 1 is not supported with stream=True")

    input_ids, prompt_len = _prepare_inputs(req)
    gen_kwargs = _build_generation_kwargs(req, prompt_len)
    created = int(time.time())
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        return _stream_response(req, input_ids, gen_kwargs, prompt_len, created, resp_id)

    def run():
        with gen_lock:
            out = model.generate(input_ids, return_dict_in_generate=True,
                                 tokenizer=tokenizer, **gen_kwargs)
        return out

    try:
        out = await asyncio.to_thread(run)
    except Exception as e:  # noqa: BLE001 - surface generation failures to the client
        raise HTTPException(500, f"generation failed: {e}") from e

    choices = []
    for seq in out.sequences:
        gen_len = seq.shape[0] - prompt_len
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        finish = "length" if gen_len >= gen_kwargs["max_new_tokens"] else "stop"
        choices.append({
            "index": len(choices),
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        })

    prompt_tokens = prompt_len
    completion_tokens = out.sequences.shape[1] - prompt_len
    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _stream_response(req: ChatCompletionRequest, input_ids, gen_kwargs, prompt_len,
                     created: int, resp_id: str):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def run():
        try:
            with gen_lock:
                model.generate(input_ids, streamer=streamer,
                               tokenizer=tokenizer, **gen_kwargs)
        except Exception as e:  # noqa: BLE001 - keep the queue from hanging forever
            streamer.text_queue.put(("__error__", e))

    threading.Thread(target=run, daemon=True).start()

    async def event_stream():
        yield _sse({
            "id": resp_id, "object": "chat.completion.chunk", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })

        content = ""
        while True:
            try:
                value = await asyncio.to_thread(streamer.text_queue.get, _QUEUE_TIMEOUT)
            except Exception:  # noqa: BLE001 - idle too long, give up
                break
            if value is None:
                break
            if isinstance(value, tuple) and value and value[0] == "__error__":
                print(f"[server] streaming generation failed: {value[1]}")
                break
            if not value:
                continue
            content += value
            yield _sse({
                "id": resp_id, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": value}, "finish_reason": None}],
            })

        completion_tokens = len(tokenizer.encode(content, add_special_tokens=False))
        yield _sse({
            "id": resp_id, "object": "chat.completion.chunk", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_len + completion_tokens,
            },
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main():
    global MODEL_ID, SHARDS_PATH, COMPRESSION, HF_TOKEN, MAX_SEQ_LEN, \
        MAX_ROUNDS, DEFAULT_MAX_NEW_TOKENS, NO_PREFETCHING

    p = argparse.ArgumentParser(description="Serve an AirLLM model behind an OpenAI-compatible API")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct")
    p.add_argument("--shards-path", default="/home/kg/airllm_shards",
                   help="directory holding the layer shards (splitted_model[.4bit])")
    p.add_argument("--compression", default="4bit", choices=[None, "4bit", "8bit"])
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                   help="context window: prompt is truncated to this many tokens and max_tokens "
                        "per request is capped at max_seq_len - prompt_len")
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                   help="keep the system prompt plus the most recent N user/assistant rounds; "
                        "older turns are dropped before the prompt is built (per-request "
                        "max_rounds overrides this)")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                   help="default max_tokens when the request does not specify one")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--no-prefetching", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    MODEL_ID = args.model
    SHARDS_PATH = args.shards_path
    COMPRESSION = args.compression
    HF_TOKEN = args.hf_token
    MAX_SEQ_LEN = args.max_seq_len
    MAX_ROUNDS = args.max_rounds
    DEFAULT_MAX_NEW_TOKENS = args.max_new_tokens
    NO_PREFETCHING = args.no_prefetching

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
