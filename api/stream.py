from api.schemas import StreamChunk, DeltaContent, StreamChoice, InferenceMetrics
import time
import json

def make_sse_chunk(token: str, chunk_id: str, model: str, is_chat: bool) -> str:
    obj_type = "chat.completion.chunk" if is_chat else "text_completion"

    chunk = StreamChunk(
        id=chunk_id,
        object=obj_type,
        created=int(time.time()),
        model=model,
        choices = [StreamChoice(delta=DeltaContent(content=token))],
    )
    return f"data: {chunk.model_dump_json()}\n\n"

def make_sse_done() -> str:
    return "data: [DONE]\n\n"

async def token_stream_generator(handle, chunk_id: str, model: str, is_chat: bool):
    async for token in handle.stream():
        yield make_sse_chunk(token, chunk_id, model, is_chat)

    # Emit metrics before [DONE]
    req = handle._request
    latency = time.time() - req.start_time
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
    yield f"data: {json.dumps({'type': 'metrics', 'metrics': metrics.model_dump()})}\n\n"
    yield make_sse_done()