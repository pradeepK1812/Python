from inspect import isabstract

import pytest

from ml.rag.retrievers.retriever import Retriever


def test_import():
    """Verify that the Retriever abstraction can be imported."""
    assert Retriever is not None


def test_retriever_is_abstract():
    """Verify that Retriever is an abstract base class."""
    assert isabstract(Retriever)

    with pytest.raises(TypeError):
        Retriever()
