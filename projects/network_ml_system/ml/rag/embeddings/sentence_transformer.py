from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel, EmbeddingVector


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """
    Implements the EmbeddingModel contract using the sentence-transformers
    library.

    The configured Sentence Transformer model is loaded during object
    construction, ensuring that the instance is fully initialized and
    ready to generate embeddings immediately after creation.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initializes the embedding model with the specified pretrained model.

        Args:
            model_name:
                Name or path of the Sentence Transformer model to load.

        Raises:
            RuntimeError:
                If the configured model cannot be loaded.
        """
        self._model_name = model_name

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Sentence Transformer model '{model_name}'."
            ) from exc

    @property
    def model_name(self) -> str:
        """
        Returns the configured embedding model name.
        """
        return self._model_name



    def embed(self, text: str) -> EmbeddingVector:
        """
        Generates a semantic embedding vector for the supplied text.

        This method delegates embedding generation to the configured
        Sentence Transformer model while conforming to the EmbeddingModel
        interface.

        Args:
            text:
                The textual content to embed.

        Returns:
            A semantic embedding vector representing the input text.

        Raises:
            RuntimeError:
                If embedding generation fails.
        """
        try:
        # Generate the embedding as a NumPy array.
            embedding_vector = self._model.encode(
              text,
            convert_to_numpy=True,)

        # Convert the NumPy array to the project's EmbeddingVector type.
            return embedding_vector.tolist()

        except Exception as exc:
              raise RuntimeError(
                f"Failed to generate embedding using "
                f"'{self._model_name}'."
              ) from exc
       
