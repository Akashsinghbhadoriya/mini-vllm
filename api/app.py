from fastapi import FastAPI
from api.routes import router
from core.server import Server


def create_app(server=None) -> FastAPI:
    app = FastAPI(title="miniVllm", version="0.1.0")

    if server is None:
        server = Server()

    app.state.server = server

    @app.on_event("startup")
    async def startup():
        server.start()

    @app.on_event("shutdown")
    async def shutdown():
        server.engine.stop()

    app.include_router(router)
    return app


app = create_app()
