# Semantic Chunking Engine

This subproject is part of the ai-core-infra modular AI infrastructure. It implements a smart chunking engine (Semantic Chunking) designed to segment long documents based on the transition of semantic meaning between sentences, optimizing data preparation for Retrieval-Augmented Generation (RAG) pipelines.

For this system an algorithm based on vector similarity was implemented, integrating directly with the local tokenizer [bpe-tokenizer-from-scratch](https://github.com/juanmmm21/bpe-tokenizer-from-scratch) to validate physical context capacity limits.

## Mathematical Foundation of Semantic Chunking

Unlike traditional fixed-size chunkers that cut text arbitrarily (potentially splitting sentences or concepts in half), the semantic chunker follows a structured, scientific methodology:

### 1. Sentence Segmentation
Plain text is split into sentences using a regex-based analyzer designed to ignore periods that are part of common Spanish and English abbreviations (e.g. `Dr.`, `Mr.`, `e.g.`, `U.S.A.`).

### 2. Contextualization via Sliding Window
To give each sentence a robust local context and avoid noise from excessively short sentences, each sentence is combined with its adjacent neighbors according to a configurable window size:

$$C_i = \text{concat}(S_{i-w}, \dots, S_i, \dots, S_{i+w})$$

Where $w$ is the sliding window size (`window_size`).

### 3. Embedding Computation and Semantic Distance
Embedding vectors are obtained for the contextualized sentences. The semantic transition is evaluated by computing the cosine distance (1 - similarity) between vectors of consecutive sentences $C_i$ and $C_{i+1}$:

$$\text{sim}(C_i, C_{i+1}) = \frac{C_i \cdot C_{i+1}}{\|C_i\|_2 \|C_{i+1}\|_2}$$

$$d_i = 1 - \text{sim}(C_i, C_{i+1})$$

### 4. Determining Dynamic Cut Points (Boundaries)
An adaptive cutoff threshold is set using the mean and standard deviation of the distance series:

$$\tau = \mu_d + k \cdot \sigma_d$$

Where:
*   $\mu_d$ is the arithmetic mean of the text's consecutive distances.
*   $\sigma_d$ is the standard deviation of the distances.
*   $k$ is the sensitivity factor (`threshold_factor`, typically in the range $1.0$ to $1.5$).

Any index $i$ where $d_i > \tau$ represents a significant topic shift and is defined as a chunk split boundary.

### 5. Physical Token Capacity Constraint
To ensure no fragment exceeds the LLM's maximum context capacity (`max_tokens`), each semantic block is pre-tokenized using the infrastructure's `BPETokenizer`. If a fragment exceeds the physical limit, a binary sentence-based subdivision is performed until the constraint is satisfied.

## Offline Embedding Provider (Mock)

For environments without internet access or GPU resources, a `MockEmbeddingProvider` class is implemented. It generates deterministic vector representations of dimension $D$ (384 by default) from character frequencies and term hashes of the text:

$$\text{vec}[i] = \sum_{w \in S} \text{hash}(w, i)$$

It is then normalized under the $L_2$ norm:

$$\text{vec}_{L2} = \frac{\text{vec}}{\|\text{vec}\|_2}$$

This preserves the validity of cosine similarity calculations, allowing testing without downloads or heavy dependencies.

## Configuration and Data Specifications

### Initialization Parameters
```python
class SemanticChunker:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        tokenizer: Optional[BPETokenizer] = None,
        window_size: int = 3,
        threshold_factor: float = 1.2,
        max_tokens: int = 512
    )
```

## Installation Requirements

*   Python 3.10 or higher
*   Numpy
*   Sentence-Transformers (optional, for real semantic embeddings)

To install the requirements, run:
```bash
pip install -r requirements.txt
```

## Execution and Verification Guide

### 1. Run Unit Tests
```bash
python3 -m unittest test_chunker.py
```

### 2. Run the Demo Script
```bash
python3 example.py
```

## Connectivity within the ai-core-infra Ecosystem

The `semantic-chunking-engine` module acts as a crucial bridge in the data pipeline:
*   Consumes [bpe-tokenizer-from-scratch](https://github.com/juanmmm21/bpe-tokenizer-from-scratch) to control the max_tokens limit.
*   In production, real embeddings can be trained locally with [contrastive-embedding-trainer](https://github.com/juanmmm21/contrastive-embedding-trainer).
*   The generated semantic fragments are optimally stored and indexed in [nano-vector-db](https://github.com/juanmmm21/nano-vector-db) and consumed by the final SPA [nexus-second-brain](https://github.com/juanmmm21/nexus-second-brain).
