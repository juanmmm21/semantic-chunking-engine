import abc
import hashlib
import random
from typing import List

class EmbeddingProvider(abc.ABC):
    """
    Interfaz abstracta para proveedores de embeddings.
    
    Permite desacoplar el motor de segmentación semántica de la biblioteca
    o API específica utilizada para calcular los vectores (por ejemplo,
    SentenceTransformers local, OpenAI, Cohere, o nuestro propio trainer).
    """
    
    @abc.abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Calcula y devuelve las representaciones vectoriales para una lista de textos.
        
        Args:
            texts: Lista de cadenas de texto.
            
        Returns:
            Lista de listas de floats representing los vectores de embeddings.
        """
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Proveedor de embeddings local que utiliza la librería SentenceTransformers.
    
    Utiliza por defecto el modelo eficiente y ligero 'all-MiniLM-L6-v2'.
    Los imports se realizan de manera perezosa (lazy) para no ralentizar la
    inicialización del módulo si esta clase no es instanciada.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """
        Carga perezosa del modelo en memoria en la primera llamada.
        """
        if self._model is None:
            # Importación local para evitar cargar PyTorch/Transformers innecesariamente
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError as e:
                raise ImportError(
                    "No se pudo importar 'sentence-transformers'. "
                    "Por favor, instala las dependencias usando 'pip install -r requirements.txt'."
                ) from e
        return self._model

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # sentence-transformers encode devuelve un array de numpy. Lo convertimos a lista de floats.
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [list(map(float, emb)) for emb in embeddings]


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Proveedor de embeddings simulado (Mock) para pruebas de integración rápidas y tests unitarios.
    
    Genera vectores de embeddings deterministas de dimensión fija (384) basados en las palabras
    del texto. Las oraciones que comparten palabras clave presentarán una alta similitud del coseno,
    simulando el comportamiento semántico real sin necesidad de descargar modelos pesados ni tener PyTorch.
    """
    
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _hash_word_to_vector(self, word: str) -> List[float]:
        """
        Genera un vector pseudoaleatorio determinista para una palabra usando su hash MD5 como semilla.
        """
        # Usamos MD5 para obtener una semilla numérica consistente a partir del string de la palabra.
        hash_digest = hashlib.md5(word.encode("utf-8")).digest()
        seed = int.from_bytes(hash_digest, byteorder="big")
        
        # Inicializamos un generador aleatorio local con esta semilla
        rng = random.Random(seed)
        # Generamos un vector con distribución normal
        return [rng.normalvariate(0.0, 1.0) for _ in range(self.dimension)]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            if not text.strip():
                # Oraciones vacías devuelven un vector de ceros
                embeddings.append([0.0] * self.dimension)
                continue
            
            # Limpiamos y dividimos en palabras en minúsculas
            words = [w.strip(",.!?()\"';:") for w in text.lower().split()]
            words = [w for w in words if len(w) > 1]
            
            if not words:
                # Si no quedan palabras legibles
                embeddings.append([0.0] * self.dimension)
                continue
            
            # Acumulamos los vectores de cada palabra para simular la composición semántica
            accum_vector = [0.0] * self.dimension
            for word in words:
                word_vector = self._hash_word_to_vector(word)
                for i in range(self.dimension):
                    accum_vector[i] += word_vector[i]
            
            # Calculamos la norma L2 para normalizar el vector resultante a longitud unitaria
            l2_norm = sum(x**2 for x in accum_vector) ** 0.5
            if l2_norm > 0:
                normalized_vector = [x / l2_norm for x in accum_vector]
            else:
                normalized_vector = [0.0] * self.dimension
                
            embeddings.append(normalized_vector)
            
        return embeddings
