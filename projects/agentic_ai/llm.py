from huggingface_hub import InferenceClient


MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


class LLM:
    """Interface to a hosted Hugging Face language model."""

    def __init__(self, model: str = MODEL):
        self.client = InferenceClient()
        self.model = model

    def generate(self, messages, tools=None):
        """Send messages to the LLM and return the complete assistant message."""

        response = self.client.chat_completion(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=512,
        )

        return response.choices[0].message
