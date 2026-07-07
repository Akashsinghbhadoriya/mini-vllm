from api.schemas import StreamChunk, DeltaContent, StreamChoice
import time

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
    yield make_sse_done()