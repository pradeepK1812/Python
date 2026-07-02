import time
import uuid
import logging

from fastapi import APIRouter, HTTPException

from app.schemas import (
    GenerationRequest,
    GenerationResponse
)

from app.services.mini_gpt_service import (
    generate_text
)

from app.config import API_VERSION

router = APIRouter()

logger = logging.getLogger(__name__)

# -------------------------
# Text Generation route
# -------------------------

@router.post(
    "/generate-text",
    response_model=GenerationResponse
)

def generate(data: GenerationRequest):

    request_id = str(uuid.uuid4())

    start_time = time.time()

    try:

        logger.info(
            f"[{request_id}] "
            f"Version={API_VERSION} "
            f"Input={data.prompt} "
            f"MaxTokens={data.max_tokens}"
        )
        generated_text = generate_text(data.prompt,data.max_tokens)
        
        elapsed_time = time.time() - start_time

        logger.info(
            f"[{request_id}] "
            f"Version={API_VERSION} "
            f"Output={generated_text}"
        )

        logger.info(
            f"[{request_id}] "
            f"Version={API_VERSION} "
            f"Completed in {elapsed_time:.4f}s"
        )

        return GenerationResponse(
               generated_text=generated_text
        )      

    except Exception as e:

        logger.error(
            f"[{request_id}] "
            f"Text Generation failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Text generation failed"
        )
