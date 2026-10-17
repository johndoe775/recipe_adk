
import os
import uvicorn

from google.adk.cli.fast_api import get_fast_api_app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALLOWED_ORIGINS = [
    "https://YOUR_USERNAME.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app = get_fast_api_app(
    agents_dir=BASE_DIR,
    allow_origins=ALLOWED_ORIGINS,
    web=False,
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
    )
