import unittest
from typing import List
from embedding_provider import MockEmbeddingProvider
from chunker import SemanticChunker

class DummyTokenizer:
    """
    Tokenizador dummy muy simple para probar el límite de tokens sin dependencias externas.
    Asume que cada carácter es aproximadamente un token o simula una longitud básica.
    """
    def encode(self, text: str) -> List[int]:
        # Cada palabra en el texto se convierte a un ID entero ficticio
        return [hash(w) for w in text.split()]


class TestSemanticChunker(unittest.TestCase):
    """
    Casos de prueba para verificar el comportamiento del segmentador semántico.
    """

    def setUp(self) -> None:
        self.mock_embeddings = MockEmbeddingProvider(dimension=64)
        self.tokenizer = DummyTokenizer()
        # Inicializamos el chunker con el proveedor mock para tests reproducibles y rápidos
        self.chunker = SemanticChunker(
            embedding_provider=self.mock_embeddings,
            tokenizer=self.tokenizer,
            window_size=3,
            threshold_factor=1.0,
            max_tokens=20
        )

    def test_split_into_sentences(self) -> None:
        """
        Prueba que la segmentación de oraciones no rompa con abreviaturas en español o inglés.
        """
        text = "El Dr. Juan trabaja en la U.S.A. como investigador. ¿Es esto increíble? ¡Sí, lo es!"
        sentences = self.chunker.split_into_sentences(text)
        
        # Deben ser 3 oraciones principales:
        # 1. El Dr. Juan trabaja en la U.S.A. como investigador
        # 2. ¿Es esto increíble?
        # 3. ¡Sí, lo es!
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].startswith("El Dr."))
        self.assertEqual(sentences[1], "¿Es esto increíble?")
        
    def test_single_sentence_or_empty(self) -> None:
        """
        Valida el comportamiento con cadenas vacías o una única oración.
        """
        # Cadena vacía
        self.assertEqual(self.chunker.chunk_text(""), [])
        
        # Una sola oración
        chunks = self.chunker.chunk_text("Solo una oración simple.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Solo una oración simple.")
        self.assertEqual(chunks[0]["index"], 0)

    def test_chunking_transitions(self) -> None:
        """
        Verifica que se produzcan divisiones semánticas claras al cambiar drásticamente de tema.
        El MockEmbeddingProvider genera vectores muy similares si comparten palabras clave,
        y muy disímiles si no comparten palabras clave.
        """
        # Texto compuesto por dos tópicos completamente distintos
        text = (
            "El telescopio James Webb observa las estrellas del espacio lejano. "
            "Las galaxias distantes brillan con luz infrarroja en el espacio. "
            "El universo profundo es capturado por cámaras del telescopio espacial. "
            # Cambio abrupto de tema a cocina:
            "Para cocinar pasta es necesario hervir agua en una olla grande. "
            "Añade sal al agua hirviendo antes de verter los fideos de pasta. "
            "La pasta se sirve caliente con salsa de tomate y queso rallado."
        )
        
        # Ajustamos el umbral para facilitar la división
        self.chunker.threshold_factor = 0.5
        chunks = self.chunker.chunk_text(text)
        
        # Deberíamos obtener al menos 2 chunks claramente diferenciados por el tema.
        self.assertTrue(len(chunks) >= 2)
        
        # Verificamos que las oraciones de astronomía queden agrupadas y separadas de la pasta.
        astro_chunk = chunks[0]["text"]
        pasta_chunk = chunks[-1]["text"]
        
        self.assertIn("telescopio", astro_chunk)
        self.assertIn("pasta", pasta_chunk)
        self.assertNotIn("pasta", astro_chunk)
        self.assertNotIn("telescopio", pasta_chunk)

    def test_token_limit_enforcement(self) -> None:
        """
        Prueba que si un chunk semántico supera max_tokens, se subdivida
        respetando el tamaño configurado.
        """
        # Cada palabra en DummyTokenizer representa un token.
        # Ponemos max_tokens a 5.
        self.chunker.max_tokens = 5
        
        # Tres oraciones con 3 palabras cada una (3 tokens cada una).
        # Total de palabras = 9. Si el límite es 5, no pueden ir todas juntas.
        text = "Sol brilla hoy. Luna sale noche. Estrellas brillan cielo."
        
        chunks = self.chunker.chunk_text(text)
        
        # Deben haberse subdividido en al menos 2 chunks para no exceder 5 tokens por chunk.
        self.assertTrue(len(chunks) >= 2)
        for chunk in chunks:
            self.assertTrue(chunk["num_tokens"] <= 5, f"Chunk excede límite: {chunk['num_tokens']} tokens")

    def test_window_size_safety(self) -> None:
        """
        Verifica que un window_size par lance una excepción al inicializar el objeto.
        """
        with self.assertRaises(ValueError):
            SemanticChunker(
                embedding_provider=self.mock_embeddings,
                window_size=4 # Ventana par no permitida
            )

if __name__ == "__main__":
    unittest.main()
