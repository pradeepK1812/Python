"""
Module: ollama.py

Defines the Ollama implementation of the LLM abstraction.

Overview
--------
OllamaLLM implements the LLM interface by communicating with a locally
running Ollama server using its REST API.

It translates textual prompts into generated text while encapsulating
all Ollama-specific communication details from the rest of the framework.

Responsibilities
----------------
- Accept textual prompts.
- Send prompt requests to the configured Ollama server.
- Validate HTTP and JSON responses.
- Extract generated text.
- Return generated text.

Guarantees
----------
- Returns generated text for successful requests.
- Uses the configured Ollama model.
- Translates provider-specific failures into framework-level exceptions.
- Never modifies the supplied prompt.

Does NOT
---------
- Construct prompts.
- Retrieve contextual knowledge.
- Generate embeddings.
- Manage conversations.
- Know about the Generator abstraction.
- Know about the RAG pipeline.

Design Principles
-----------------
- Infrastructure layer component.
- Concrete implementation of the LLM abstraction.
- Encapsulates all Ollama-specific communication.
- Independent of Generator and prompt construction.

--------------------------------------------------
Transformation
--------------------------------------------------

Input:
    str (Prompt)

Output:
    str (Generated text)

--------------------------------------------------
Dependencies
--------------------------------------------------

Depends on:
- requests
- Python standard library

Does not depend on the Generator or other business components.
"""


from __future__ import annotations
import requests
from .base import LLM
from requests import Response
from typing import Any
from typing import override

class OllamaLLM(LLM):
    """
    Concrete implementation of the LLM abstraction using Ollama.

    Generates text by communicating with a locally running Ollama
    server through its REST API while encapsulating all provider-
    specific communication details.
    """

    def __init__(
    self,
    model_name: str,
    host: str = "http://localhost:11434",
    timeout: float = 180.0,    #default timeout of OLLAMa connection
    ) -> None:

        model_name = model_name.strip()

        if not model_name:
           raise ValueError("model_name must not be empty.")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self._model_name = model_name
        self._host = host.rstrip("/")
        self._timeout = timeout

    @property
    @override
    def model_name(self) -> str:
       """
       Returns the configured Ollama model name.
       """
       return self._model_name
    
    @override   # Optional in Python 3.12 (typing.override)
    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text for the supplied prompt using the configured
        Ollama model.

        Args:
            prompt:
                The textual prompt supplied to the language model.

        Returns:
            The generated text.

        Raises:
            RuntimeError:
                If text generation fails or an invalid response is
                received from the Ollama server.
        """

        # ------------------------------------------------------------------
        # Build the Ollama generate endpoint URL.
        # ------------------------------------------------------------------
        url = f"{self._host}/api/generate"

        #------------------------------------------------
        # Build request payload.
        #------------------------------------------------

        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
        }

        #---------------------------------------------------
        # Send HTTP request.
        #---------------------------------------------------

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self._timeout,
            )

            response.raise_for_status()
        
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Unable to connect to the Ollama server."
            ) from exc

        except requests.Timeout as exc:
            raise RuntimeError(
                "Request to Ollama timed out."
            ) from exc

        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed: HTTP {response.status_code}."
            ) from exc


        # ------------------------------------------------------------------
        # Parse the JSON response.
        # ------------------------------------------------------------------
        try:
            response_json = response.json()

        except ValueError as exc:
            raise RuntimeError(
                "Invalid JSON received from the Ollama server."
            ) from exc


        # ------------------------------------------------------------------
        # Extract the generated text from the response.
        # ------------------------------------------------------------------
        try:
            generated_text = response_json["response"]

        except KeyError as exc:
            raise RuntimeError(
                "Malformed response received from the Ollama server."
            ) from exc


        # ------------------------------------------------------------------
        # Return the generated text.
        # ------------------------------------------------------------------
        return generated_text
