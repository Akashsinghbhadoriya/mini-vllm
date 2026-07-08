from fastapi import APIRouter, Request as FastAPIRequest, Depends
from fastapi.responses import StreamingResponse
import asyncio
import time
import uuid

from api.schemas import (
    CompletionRequest, CompletionResponse, CompletionChoice,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    ChatCompletionMessage, UsageInfo, InferenceMetrics
)
from api.stream import token_stream_generator

router = APIRouter()


def get_server(request: FastAPIRequest):
    return request.app.state.server


def messages_to_prompt(messages) -> str:
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"[System]: {msg.content}")
        elif msg.role == "user":
            parts.append(f"[User]: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"[Assistant]: {msg.content}")
    parts.append("[Assistant]:")
    return "\n".join(parts)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/metrics")
async def metrics(server=Depends(get_server)):
    return {
        "active_requests": len(server.scheduler.active_requests),
        "queue_size": server.request_queue.size(),
        "total_requests": server.request_counter,
    }


@router.post("/v1/completions")
async def completions(body: CompletionRequest, server=Depends(get_server)):
    handle = server.submit(body.prompt, body.max_tokens, body.stream)
    chunk_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    if not body.stream:
        loop = asyncio.get_event_loop()
        text, req_id, latency = await loop.run_in_executor(None, handle.wait)
        req = handle._request
        prompt_tokens = len(req.prompt_token_ids or [])
        completion_tokens = len(req.generated_token_ids)
        ttft_ms = ((req.first_token_time or time.time()) - req.start_time) * 1000
        metrics = InferenceMetrics(
            ttft_ms=round(ttft_ms, 1),
            latency_ms=round(latency * 1000, 1),
            tokens_per_sec=round(completion_tokens / latency, 1) if latency > 0 else 0.0,
            prefix_cache="HIT" if req.cached_prefix_len > 0 else "MISS",
            kv_blocks=req.block_table.num_blocks(),
            batch_id=req.request_id,
        )
        return CompletionResponse(
            id=chunk_id,
            created=int(time.time()),
            model=body.model,
            choices=[CompletionChoice(text=text)],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metrics=metrics,
        )

    generator = token_stream_generator(handle, chunk_id, body.model, is_chat=False)
    return StreamingResponse(generator, media_type="text/event-stream")


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, server=Depends(get_server)):
    prompt = messages_to_prompt(body.messages)
    handle = server.submit(prompt, body.max_tokens, body.stream)
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if not body.stream:
        loop = asyncio.get_event_loop()
        text, req_id, latency = await loop.run_in_executor(None, handle.wait)
        req = handle._request
        prompt_tokens = len(req.prompt_token_ids or [])
        completion_tokens = len(req.generated_token_ids)
        ttft_ms = ((req.first_token_time or time.time()) - req.start_time) * 1000
        metrics = InferenceMetrics(
            ttft_ms=round(ttft_ms, 1),
            latency_ms=round(latency * 1000, 1),
            tokens_per_sec=round(completion_tokens / latency, 1) if latency > 0 else 0.0,
            prefix_cache="HIT" if req.cached_prefix_len > 0 else "MISS",
            kv_blocks=req.block_table.num_blocks(),
            batch_id=req.request_id,
        )
        return ChatCompletionResponse(
            id=chunk_id,
            created=int(time.time()),
            model=body.model,
            choices=[ChatCompletionChoice(
                message=ChatCompletionMessage(content=text)
            )],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metrics=metrics,
        )

    generator = token_stream_generator(handle, chunk_id, body.model, is_chat=True)
    return StreamingResponse(generator, media_type="text/event-stream")
