import os
import sys
from typing import List

# Interlinking: Importamos de forma dinámica el BPETokenizer propio de la infraestructura
# agregando su ruta local al sys.path para cumplir las directrices de reutilización de código.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bpe-tokenizer-from-scratch")))

try:
    from tokenizer import BPETokenizer
except ImportError:
    # Fallback por si la carpeta no está en la misma estructura relativa
    BPETokenizer = None

from embedding_provider import LocalEmbeddingProvider, MockEmbeddingProvider
from chunker import SemanticChunker

def get_bpe_tokenizer() -> BPETokenizer:
    """
    Instancia y entrena rápidamente el BPETokenizer de la infraestructura
    para que funcione como contador de tokens nativo.
    """
    if BPETokenizer is None:
        print("Advertencia: No se encontró el módulo BPETokenizer propio. Se utilizará un contador básico de palabras.")
        return None
        
    print("Inicializando y entrenando el BPETokenizer propio de la infraestructura...")
    tokenizer = BPETokenizer()
    # Entrenamos con un corpus de calibración corto para inicializar sus reglas
    calibration_corpus = (
        "El procesamiento de lenguaje y los embeddings semánticos permiten "
        "estructurar textos largos. La programación y el entrenamiento de modelos "
        "son áreas complejas de la inteligencia artificial moderna."
    )
    tokenizer.train(calibration_corpus, vocab_size=280)
    return tokenizer


def run_demo() -> None:
    # 1. Texto multitópico que simula un documento real con cambios drásticos de tema:
    # - Tópico A: Computación Cuántica (Oraciones 1-3)
    # - Tópico B: Gastronomía/Cocina (Oraciones 4-6)
    # - Tópico C: Programación/Software (Oraciones 7-9)
    multitopic_text = (
        "La computación cuántica es un paradigma de computación distinto al clásico. "
        "Utiliza cúbits en lugar de bits tradicionales para realizar cálculos complejos. "
        "Los fenómenos de superposición y entrelazamiento permiten procesar información en paralelo. "
        # Transición a cocina
        "Para preparar una buena paella valenciana es clave sofreír bien los ingredientes. "
        "El arroz debe cocinarse a fuego lento en un caldo de pollo y verduras sabroso. "
        "El azafrán le aporta a la paella su color amarillo característico y aroma único. "
        # Transición a programación
        "El lenguaje de programación Python destaca por su sintaxis limpia y legible. "
        "Es ampliamente utilizado en ciencia de datos, desarrollo web y automatización. "
        "Escribir código limpio y documentado facilita enormemente el mantenimiento del software."
    )

    print("=== PASO 1: Configuración del Proveedor de Embeddings ===")
    # Intentamos inicializar el proveedor local de SentenceTransformers.
    # Si no está instalado, caemos automáticamente en el proveedor Mock
    # para garantizar que la demo se ejecute sin requerir descargar GB de dependencias si no se desea.
    try:
        print("Intentando cargar LocalEmbeddingProvider (sentence-transformers)...")
        provider = LocalEmbeddingProvider()
        # Forzamos la inicialización llamando a la propiedad lazy
        _ = provider.model
        print("✓ Proveedor local cargado con éxito.")
    except Exception as e:
        print(f"No se pudo cargar el proveedor local ({e}).")
        print("Usando MockEmbeddingProvider (simulación de similitud semántica por hash de palabras).")
        provider = MockEmbeddingProvider()
    print("-" * 70)

    print("\n=== PASO 2: Inicialización del Tokenizador Propio ===")
    tokenizer = get_bpe_tokenizer()
    print("-" * 70)

    print("\n=== PASO 3: Ejecución del Semantic Chunker ===")
    # Inicializamos el segmentador semántico.
    # Usamos un max_tokens bajo (ej. 30 tokens) para demostrar cómo subdivide chunks semánticos
    # que son demasiado largos, garantizando que respeten la restricción.
    chunker = SemanticChunker(
        embedding_provider=provider,
        tokenizer=tokenizer,
        window_size=3,
        threshold_factor=0.8,
        max_tokens=150
    )

    print(f"Ejecutando fragmentación con max_tokens={chunker.max_tokens}...")
    chunks = chunker.chunk_text(multitopic_text)
    print(f"¡Texto fragmentado en {len(chunks)} chunks!")
    print("-" * 70)

    print("\n=== PASO 4: Visualización de los Chunks Generados ===")
    for chunk in chunks:
        print(f"\n[CHUNK {chunk['index']}]")
        print(f"  Tokens estimados (BPE Propio): {chunk['num_tokens']}")
        print(f"  Número de oraciones: {len(chunk['sentences'])}")
        print(f"  Contenido:")
        print(f"    \"{chunk['text']}\"")
        print("  " + "."*50)

    print("\n✓ ¡Demostración interactiva completada correctamente!")
    print("-" * 70)

if __name__ == "__main__":
    run_demo()
