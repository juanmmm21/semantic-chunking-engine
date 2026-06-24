import re
import math
from typing import List, Dict, Any, Optional, Protocol
from embedding_provider import EmbeddingProvider

class TokenizerProtocol(Protocol):
    """
    Protocolo estructural (Duck Typing) que define la interfaz mínima
    que debe cumplir cualquier tokenizador inyectado en el SemanticChunker.
    Esto permite usar nuestro BPETokenizer propio sin acoplamientos rígidos.
    """
    def encode(self, text: str) -> List[int]:
        ...


class SemanticChunker:
    """
    Segmentador de documentos inteligente basado en variaciones semánticas.
    
    Analiza el flujo de información de un texto dividiéndolo en oraciones, calculando
    la similitud del coseno entre oraciones adyacentes en base a sus embeddings,
    e identificando límites de corte donde se produce un cambio abrupto de significado.
    Adicionalmente, limita el tamaño de los chunks resultantes controlando la cantidad
    de tokens mediante un tokenizador propio.
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        tokenizer: Optional[TokenizerProtocol] = None,
        window_size: int = 3,
        threshold_factor: float = 1.2,
        max_tokens: int = 512
    ) -> None:
        """
        Args:
            embedding_provider: Instancia que calcula los embeddings.
            tokenizer: Tokenizador para contar tokens en cada chunk (opcional).
            window_size: Tamaño de la ventana deslizante (debe ser impar) para contextualizar oraciones.
            threshold_factor: Multiplicador de desviación estándar para establecer el umbral dinámico de corte.
            max_tokens: Límite estricto de tokens permitidos por cada chunk.
        """
        self.embedding_provider = embedding_provider
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.max_tokens = max_tokens
        
        if window_size % 2 == 0:
            raise ValueError("El tamaño de la ventana (window_size) debe ser un número impar para ser simétrico.")

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Divide un texto largo en una lista de oraciones individuales utilizando expresiones regulares.
        Evita cortar en abreviaturas comunes en español e inglés, así como en siglas e iniciales.
        Conserva los signos de puntuación finales correspondientes a cada oración.
        """
        if not text:
            return []
            
        # Para evitar problemas con look-behinds de longitud variable, enmascaramos
        # los puntos de las abreviaturas conocidas utilizando un marcador exclusivo.
        placeholder = "___ABBR_DOT___"
        abbreviations = ["Sr.", "Dr.", "Lic.", "Dra.", "Dña.", "Mrs.", "Mr.", "Ms.", "etc.", "p.ej.", "e.g."]
        
        temp_text = text
        for abbr in abbreviations:
            pattern = re.compile(r'\b' + re.escape(abbr), re.IGNORECASE)
            temp_text = pattern.sub(abbr.replace(".", placeholder), temp_text)
            
        # También enmascaramos iniciales de nombres y siglas de una sola letra seguidas de punto (ej. U.S.A.)
        # para que no provoquen rupturas de oración internas.
        pattern_initials = re.compile(r'\b([A-Za-z])\.')
        temp_text = pattern_initials.sub(r'\1' + placeholder, temp_text)
        
        # Dividimos utilizando un split basado en espacios precedidos por un punto, signo
        # de interrogación o exclamación. Esto preserva la puntuación final de cada oración.
        sentence_endings = re.compile(r'(?<=\.|\?|!)\s+')
        raw_splits = sentence_endings.split(temp_text)
        
        sentences = []
        for s in raw_splits:
            if s:
                cleaned = s.strip()
                if cleaned:
                    # Restauramos los puntos enmascarados de abreviaturas e iniciales.
                    restored = cleaned.replace(placeholder, ".")
                    sentences.append(restored)
                    
        return sentences

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calcula la similitud de coseno entre dos vectores numéricos.
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a ** 2 for a in vec_a) ** 0.5
        norm_b = sum(b ** 2 for b in vec_b) ** 0.5
        
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _get_contextual_sentences(self, sentences: List[str]) -> List[str]:
        """
        Aplica una ventana deslizante alrededor de cada oración, concatenando
        las oraciones vecinas. Esto proporciona un contexto semántico local
        que estabiliza los embeddings de oraciones individuales cortas.
        """
        contextual_sentences = []
        half_window = self.window_size // 2
        
        for i in range(len(sentences)):
            start = max(0, i - half_window)
            end = min(len(sentences), i + half_window + 1)
            # Concatenamos las oraciones dentro de la ventana del índice actual
            context_text = " ".join(sentences[start:end])
            contextual_sentences.append(context_text)
            
        return contextual_sentences

    def _find_boundaries(self, distances: List[float]) -> List[int]:
        """
        Calcula el umbral dinámico de distancia semántica y devuelve los índices
        donde la distancia supera dicho umbral.
        
        Fórmula del umbral: media(distancias) + factor * desv_estandar(distancias)
        """
        if not distances:
            return []
            
        mean_val = sum(distances) / len(distances)
        
        # Cálculo de la desviación estándar
        variance = sum((d - mean_val) ** 2 for d in distances) / len(distances)
        std_val = math.sqrt(variance)
        
        threshold = mean_val + (self.threshold_factor * std_val)
        
        # Los índices resultantes corresponden a los puntos de corte.
        # Un índice 'i' indica que hay un corte entre la oración 'i' y la oración 'i+1'.
        boundaries = [i for i, d in enumerate(distances) if d > threshold]
        return boundaries

    def _enforce_token_limit(self, chunk_sentences: List[str]) -> List[List[str]]:
        """
        Verifica si un fragmento de oraciones supera el número máximo de tokens (max_tokens).
        Si lo supera, lo subdivide de forma secuencial respetando los límites de las oraciones.
        """
        if not self.tokenizer:
            return [chunk_sentences]
            
        sub_chunks: List[List[str]] = []
        current_sub_chunk: List[str] = []
        current_tokens = 0
        
        for sentence in chunk_sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # Si una única oración supera por sí misma max_tokens, la agregamos
            # en su propio chunk para evitar bucles infinitos.
            if sentence_tokens > self.max_tokens:
                if current_sub_chunk:
                    sub_chunks.append(current_sub_chunk)
                    current_sub_chunk = []
                    current_tokens = 0
                sub_chunks.append([sentence])
                continue
                
            # Si al añadir la oración actual superamos el límite, cerramos el sub-chunk.
            if current_tokens + sentence_tokens > self.max_tokens:
                sub_chunks.append(current_sub_chunk)
                current_sub_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sub_chunk.append(sentence)
                current_tokens += sentence_tokens
                
        if current_sub_chunk:
            sub_chunks.append(current_sub_chunk)
            
        return sub_chunks

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Fragmenta semánticamente un texto largo.
        
        Devuelve una lista de diccionarios, donde cada diccionario contiene:
            - 'text': El contenido de texto del chunk.
            - 'sentences': Lista de las oraciones que lo integran.
            - 'num_tokens': Cantidad de tokens estimados (si hay tokenizador configurado).
            - 'index': Posición secuencial del chunk.
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
            
        # Si hay menos oraciones que el umbral de vecindario mínimo, retornamos un único chunk
        if len(sentences) <= 1:
            chunk_text_content = sentences[0]
            tokens_count = len(self.tokenizer.encode(chunk_text_content)) if self.tokenizer else None
            return [{
                "text": chunk_text_content,
                "sentences": sentences,
                "num_tokens": tokens_count,
                "index": 0
            }]

        # 1. Obtener oraciones contextualizadas y calcular sus embeddings
        contextual_sents = self._get_contextual_sentences(sentences)
        embeddings = self.embedding_provider.get_embeddings(contextual_sents)
        
        # 2. Calcular distancias semánticas (1 - similitud de coseno) entre oraciones contiguas
        distances: List[float] = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            distances.append(1.0 - sim)
            
        # 3. Identificar puntos de corte basados en el umbral dinámico
        boundaries = self._find_boundaries(distances)
        
        # 4. Agrupar las oraciones originales en base a los límites detectados
        raw_chunks: List[List[str]] = []
        current_chunk: List[str] = []
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            if i in boundaries:
                raw_chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            raw_chunks.append(current_chunk)
            
        # 5. Aplicar la restricción del límite de tokens (si está activo el tokenizador)
        final_chunks: List[Dict[str, Any]] = []
        chunk_index = 0
        
        for raw_chunk in raw_chunks:
            # Subdividir el fragmento semántico si excede max_tokens
            sub_divided = self._enforce_token_limit(raw_chunk)
            
            for sub_chunk_sents in sub_divided:
                chunk_text_content = " ".join(sub_chunk_sents)
                tokens_count = len(self.tokenizer.encode(chunk_text_content)) if self.tokenizer else None
                
                final_chunks.append({
                    "text": chunk_text_content,
                    "sentences": sub_chunk_sents,
                    "num_tokens": tokens_count,
                    "index": chunk_index
                })
                chunk_index += 1
                
        return final_chunks
