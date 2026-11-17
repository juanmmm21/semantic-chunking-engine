"""Compatibilidad: el código vive en el paquete `semantic_chunking_engine`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `chunker`
directamente."""

from semantic_chunking_engine.chunker import *  # noqa: F401,F403
from semantic_chunking_engine.chunker import SemanticChunker, TokenizerProtocol  # noqa: F401
