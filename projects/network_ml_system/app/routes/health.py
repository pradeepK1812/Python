from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.config import API_VERSION


#"version": API_VERSION

router = APIRouter()


@router.get("/", response_class=HTMLResponse)

def home():
    return """
    <html>
        <head>
            <title>Network ML API</title>
            <style>
                body {
                    font-family: Arial;
                    padding: 40px;
                    background-color: #f4f4f4;
                }

                .container {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    max-width: 600px;
                }

                a {
                    color: blue;
                    text-decoration: none;
                }
            </style>
        </head>

        <body>
            <div class="container">
                <h1>🚀 Network ML API</h1>
                <p>API server is running successfully.</p>

                <p>
                    Open
                    <a href="/docs">/docs</a>
                    for Swagger UI documentation.
                </p>
            </div>
        </body>
    </html>
    """

@router.get("/health")
def health():
    return {
        "service": "Network ML API",
        "version": "API_VERSION",
        "status": "running"
    }
