# Semantic Chunking Engine

Este subproyecto forma parte de la infraestructura modular de Inteligencia Artificial ai-core-infra. Implementa un motor de fragmentacion inteligente (Semantic Chunking) disenado para segmentar documentos extensos en base a la transicion del significado semantico entre oraciones, optimizando la preparacion de datos para pipelines de Generacion Aumentada por Recuperacion (RAG).

Para este sistema se implemento un algoritmo basado en similitud vectorial que se integra directamente con el tokenizador local [bpe-tokenizer-from-scratch](https://github.com/juanmmm21/bpe-tokenizer-from-scratch) para validar los limites de capacidad fisica de contexto.

## Fundamento Matematico del Chunking Semantico

A diferencia de los fragmentadores tradicionales de tamano fijo que cortan el texto de forma arbitraria (pudiendo dividir oraciones o conceptos a la mitad), el fragmentador semantico sigue una metodologia estructurada y cientifica:

### 1. Segmentacion de Oraciones
El texto plano se divide en oraciones utilizando un analizador basado en expresiones regulares disenado para ignorar puntos que forman parte de abreviaturas comunes en espanol e ingles (por ejemplo: `Dr.`, `Sr.`, `p.ej.`, `U.S.A.`).

### 2. Contextualizacion por Ventana Deslizante (Sliding Window)
Para dotar a cada oracion de un contexto local robusto y evitar el ruido de oraciones excesivamente cortas, cada oracion se combina con sus vecinas adyacentes segun un tamano de ventana parametrizable:

$$C_i = \text{concat}(S_{i-w}, \dots, S_i, \dots, S_{i+w})$$

Donde $w$ es el tamano de la ventana deslizante (`window_size`).

### 3. Calculo de Embeddings y Distancia Semantica
Se obtienen los vectores de embeddings de las oraciones contextualizadas. La transicion semantica se evalua calculando la distancia de coseno (1 - similitud) entre vectores de oraciones consecutivas $C_i$ y $C_{i+1}$:

$$\text{sim}(C_i, C_{i+1}) = \frac{C_i \cdot C_{i+1}}{\|C_i\|_2 \|C_{i+1}\|_2}$$

$$d_i = 1 - \text{sim}(C_i, C_{i+1})$$

### 4. Determinacion de Puntos de Corte Dinamicos (Boundaries)
Se establece un umbral de corte adaptativo utilizando la media y desviacion estandar de la serie de distancias:

$$\tau = \mu_d + k \cdot \sigma_d$$

Donde:
*   $\mu_d$ es el promedio aritmetico de las distancias consecutivas del texto.
*   $\sigma_d$ es la desviacion estandar de las distancias.
*   $k$ es el factor de sensibilidad (`threshold_factor`, tipicamente en el rango de $1.0$ a $1.5$).

Cualquier indice $i$ donde $d_i > \tau$ representa un salto tematico significativo y se define como un limite de division de chunk.

### 5. Restriccion de Capacidad Fisica de Tokens
Para asegurar que ningun fragmento exceda la capacidad maxima de contexto del LLM (`max_tokens`), cada bloque semantico se pre-tokeniza utilizando el `BPETokenizer` de la infraestructura. Si un fragmento excede el limite fisico, se realiza una subdivision binaria basada en oraciones hasta cumplir con la restriccion.

## Proveedor de Embeddings Fuera de Linea (Mock)

Para entornos donde no se disponga de conexion a internet o de recursos GPU, se implementa una clase `MockEmbeddingProvider`. Esta genera representaciones vectoriales deterministicas de dimension $D$ (por defecto 384) a partir de las frecuencias de caracteres y hashes de terminos del texto:

$$\text{vec}[i] = \sum_{w \in S} \text{hash}(w, i)$$

Posteriormente, se normaliza bajo la norma $L_2$:

$$\text{vec}_{L2} = \frac{\text{vec}}{\|\text{vec}\|_2}$$

Esto preserva la validez de los calculos de similitud de coseno, permitiendo pruebas sin descargas ni dependencias pesadas.

## Especificaciones de Configuracion y Datos

### Parametros de Inicializacion
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

## Requisitos de Instalacion

*   Python 3.10 o superior
*   Numpy
*   Sentence-Transformers (opcional, para embeddings semanticos reales)

Para instalar los requisitos, ejecute:
```bash
pip install -r requirements.txt
```

## Guia de Ejecucion y Verificacion

### 1. Ejecutar Pruebas Unitarias
```bash
python3 -m unittest test_chunker.py
```

### 2. Ejecutar Script Demostrativo
```bash
python3 example.py
```

## Conectividad en el Ecosistema ai-core-infra

El modulo `semantic-chunking-engine` actua como un puente crucial en el pipeline de datos:
*   Consume a [bpe-tokenizer-from-scratch](https://github.com/juanmmm21/bpe-tokenizer-from-scratch) para el control del limite max_tokens.
*   En produccion, los embeddings reales pueden ser entrenados localmente con [contrastive-embedding-trainer](https://github.com/juanmmm21/contrastive-embedding-trainer).
*   Los fragmentos semanticos generados son almacenados e indexados de forma optima en [nano-vector-db](https://github.com/juanmmm21/nano-vector-db) y consumidos por la SPA final [nexus-second-brain](https://github.com/juanmmm21/nexus-second-brain).
