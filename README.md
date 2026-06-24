# Semantic Chunking Engine

Este subproyecto forma parte de la infraestructura modular de Inteligencia Artificial ai-core-infra. Implementa un motor de fragmentación inteligente (Semantic Chunking) diseñado para segmentar documentos extensos en base a la transición del significado semántico entre oraciones, optimizando la preparación de datos para pipelines de Generación Aumentada por Recuperación (RAG).

Para este sistema decidí no utilizar soluciones comerciales ni fragmentadores estáticos por longitud de caracteres; en su lugar, implementé un algoritmo basado en similitud vectorial que se integra directamente con mi propio tokenizador [bpe-tokenizer-from-scratch](https://github.com/juanmmm21/bpe-tokenizer-from-scratch) para validar los límites de capacidad física de contexto.

---

## Arquitectura del Fragmentador

A diferencia de los fragmentadores tradicionales de tamaño fijo que cortan el texto de forma arbitraria (pudiendo dividir oraciones o conceptos a la mitad), el fragmentador semántico sigue una metodología estructurada:

1. **Segmentación de Oraciones:** El texto plano se divide en oraciones individuales utilizando un analizador por expresiones regulares que respeta abreviaturas comunes de uso frecuente en español e inglés (ej. Dr., Sr., etc., p.ej., U.S.A.).
2. **Contextualización por Ventana Deslizante (Sliding Window):** Para dotar a cada oración de un contexto local robusto y evitar el ruido de oraciones excesivamente cortas, cada oración se combina con sus vecinas adyacentes según un tamaño de ventana configurable (ej. window_size = 3).
3. **Cálculo de Embeddings y Distancia Semántica:** Se obtienen los vectores de embeddings de las oraciones contextualizadas. La transición semántica se evalúa calculando la distancia del coseno (1 - similitud) entre oraciones consecutivas.
4. **Puntos de Corte Dinámicos (Boundaries):** Se establece un umbral de corte adaptativo utilizando la fórmula matemática:
   $$\text{Umbral} = \text{Media}(\text{Distancias}) + \text{factor-umbral} \times \text{Desviación Estándar}(\text{Distancias})$$
   Cualquier distancia consecutiva que supere este umbral denota una ruptura temática y se marca como límite de chunk.
5. **Restricción de Tokens Integrada (Interlinking):** Para asegurar que ningún fragmento exceda la ventana máxima del LLM (max_tokens), cada bloque semántico se pre-tokeniza utilizando el BPETokenizer de mi primer proyecto. Si un bloque es excesivamente largo, se divide de manera secuencial a nivel de oraciones.

---

## Tecnologías Utilizadas

- **Python 3.10+** (Implementación orientada a objetos con tipado estricto).
- **Numpy:** Para álgebra lineal, cálculo de la desviación estándar y operaciones vectoriales de similitud del coseno de alta eficiencia.
- **Sentence-Transformers (Hugging Face):** Implementación de embeddings locales basada en PyTorch (modelo por defecto: all-MiniLM-L6-v2).
- **BPETokenizer Propio:** Integración local nativa mediante inyección de dependencias para validación y conteo de tokens.

---

## Instalación y Uso

### 1. Clonar e Inicializar
Clona este repositorio en tu máquina local y accede al directorio del proyecto:
```bash
git clone https://github.com/juanmmm21/semantic-chunking-engine.git
cd semantic-chunking-engine
```

### 2. Instalar Dependencias
Se recomienda utilizar un entorno virtual de Python. Puedes instalar los requisitos matemáticos y de machine learning mediante:
```bash
pip install -r requirements.txt
```

### 3. Ejecutar Ejemplo de Demostración
El archivo `example.py` demuestra el ciclo completo:
- Carga e inicializa el BPETokenizer del submódulo vecino.
- Configura el proveedor de embeddings. Si no detecta la librería sentence-transformers instalada, el programa cae de forma segura en un proveedor simulado (MockEmbeddingProvider) que genera representaciones consistentes basadas en hash de palabras para que la demo funcione sin dependencias complejas.
- Segmenta un documento en español con transiciones de tema muy evidentes (computación cuántica, gastronomía y desarrollo de software).
- Imprime detalladamente cada chunk resultante y su correspondiente cantidad de tokens estimada.

Ejecútalo con:
```bash
python3 example.py
```

### 4. Ejecutar Pruebas Unitarias
El proyecto cuenta con cobertura de pruebas unitarias que validan la segmentación de oraciones, límites de tokens y la lógica del umbral de desviación estándar:
```bash
python3 -m unittest test_chunker.py
```

---

## Conexión con el Ecosistema ai-core-infra

El semantic-chunking-engine desempeña un rol de ingesta crucial:
- Consume directamente las capacidades de conteo de [bpe-tokenizer-from-scratch](https://github.com/juanmmm21/bpe-tokenizer-from-scratch) para respetar la ventana de contexto.
- En fases posteriores de la infraestructura, los embeddings serán generados por mi propio modelo desarrollado en [contrastive-embedding-trainer](https://github.com/juanmmm21/contrastive-embedding-trainer).
- Los chunks limpios resultantes serán indexados y consultados en mi base de datos vectorial [nano-vector-db](https://github.com/juanmmm21/nano-vector-db) como parte del pipeline de recuperación para la aplicación final [nexus-second-brain](https://github.com/juanmmm21/nexus-second-brain).
