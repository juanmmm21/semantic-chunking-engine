"""Compatibilidad: el código vive en el paquete `semantic_chunking_engine`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `embedding_provider`
directamente."""

from semantic_chunking_engine.embedding_provider import *  # noqa: F401,F403
from semantic_chunking_engine.embedding_provider import EmbeddingProvider, LocalEmbeddingProvider, MockEmbeddingProvider  # noqa: F401
