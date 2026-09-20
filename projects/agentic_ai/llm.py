from huggingface_hub import InferenceClient
from abc import ABC, abstractmethod
import os
from groq import Groq

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


class LLM(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    def generate(self, messages, tools=None):
        """Generate an assistant response."""
        pass


class HuggingFaceLLM(LLM):
    """Hugging Face hosted inference implementation."""

    def __init__(self, model: str = MODEL):
        self.client = InferenceClient()
        self.model = model

    def generate(self, messages, tools=None):
        """Send messages to the Hugging Face model."""

        response = self.client.chat_completion(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=2048,
        )

        return response.choices[0].message


class GroqLLM(LLM):
    """Groq hosted inference implementation."""

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.client = Groq(
            api_key=os.environ["GROQ_API_KEY"]
        )
        self.model = model

    def generate(self, messages, tools=None):
        """Send messages to the Groq model."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=512,
        )

        return response.choices[0].message
