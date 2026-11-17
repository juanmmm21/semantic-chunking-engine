"""Semantic chunking by embedding-distance breakpoints with token limits."""

from .embedding_provider import EmbeddingProvider, LocalEmbeddingProvider, MockEmbeddingProvider
from .chunker import SemanticChunker, TokenizerProtocol

__version__ = "0.1.0"

__all__ = ['EmbeddingProvider', 'LocalEmbeddingProvider', 'MockEmbeddingProvider', 'SemanticChunker', 'TokenizerProtocol']
