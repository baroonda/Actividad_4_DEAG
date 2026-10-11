# Benchmark y Selección de Modelos Preentrenados de Hugging Face para Análisis de Sentimientos

## 1. Descripción del Caso de Uso
* **Contexto de Negocio**: Implementación de un componente analítico automatizado para procesar retroalimentación crítica y reseñas textuales de usuarios (crítica cinematográfica/plataformas de streaming) clasificando la polaridad en dos vertientes operativas: Positivo (`POS`) y Negativo (`NEG`).
* **Objetivo de IA**: Evaluar modelos preentrenados del Hugging Face Hub bajo condiciones homogéneas de hardware para identificar la arquitectura que maximice el equilibrio entre exactitud de clasificación y baja latencia de inferencia en entornos de producción.
* **Dominio**: Procesamiento de Lenguaje Natural (NLP - Text Classification).

## 2. Ecosistema Hugging Face y Criterios de Selección
* **Componentes del Ecosistema Hugging Face Utilizados**:
  * **Hugging Face Hub / Models**: Repositorio central de donde se descargaron los pesos preentrenados y tokenizadores de arquitecturas basadas en Transformers.
  * **Hugging Face Datasets**: Biblioteca de ingesta estructurada empleada para descargar, particionar y muestrear el corpus de evaluación sin requerir descargas manuales dispersas.
  * **Hugging Face Evaluate & Pipelines**: Abstracción de alto nivel (`transformers.pipeline`) para orquestar la tokenización, inferencia en GPU y posprocesamiento determinístico.
* **Criterios Técnicos y Prácticos de Selección**:
  1. *Especialización en la tarea*: Modelos afinados específicamente en tareas de clasificación de polaridad y sentimiento en texto.
  2. *Diversidad arquitectónica y peso*: Selección de un modelo ligero y destilado (DistilBERT), un modelo optimizado en redes sociales (Twitter RoBERTa) y un transformador multilingüe de gran escala (mBERT).
  3. *Licenciamiento*: Modelos con licencias abiertas permisivas (Apache 2.0 y MIT) que garantizan su uso y adaptación técnica.

## 3. Dataset y Soporte Reproducible
* **Origen del Dataset**: Subconjunto de prueba del corpus público estandarizado **Stanford IMDB** (`stanfordnlp/imdb`), obtenido mediante la librería `datasets` de Hugging Face.
* **Estructura de Datos**:
  * `text` (*string*): Fragmento textual correspondiente a la reseña del usuario.
  * `label` (*string*): Etiqueta supervisada binaria (`POS`, `NEG`).
* **Procedimiento de Muestreo y Balanceo**:
  * Se seleccionó una muestra estratificada de 150 ejemplos (75 positivos y 75 negativos) con semilla aleatoria controlada (`random_state=42`) para garantizar reproducibilidad exacta y mitigar sesgos de clase mayoritaria.
  * El conjunto se exportó y preserva localmente en la ruta `data/reviews_dataset.csv`.

## 4. Reporte Técnico Comparativo

### Tabla Comparativa de Modelos
| Modelo | Hugging Face ID | Tarea | Tamaño / Parámetros | Licencia | Accuracy | Precision | Recall | F1-Score | Latencia Promedio (GPU) | Recursos Computacionales |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DistilBERT SST-2** | `distilbert-base-uncased-finetuned-sst-2-english` | `text-classification` | ~66M (~268 MB) | Apache 2.0 | **0.8467** | **0.8522** | **0.8467** | **0.8461** | **~8.5 ms** | Muy bajo (ideal para inferencia masiva/CPU) |
| **Twitter RoBERTa** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `text-classification` | ~125M (~500 MB) | MIT | 0.8133 | 0.8245 | 0.8133 | 0.8120 | ~14.8 ms | Medio |
| **Multilingual BERT** | `nlptown/bert-base-multilingual-uncased-sentiment` | `text-classification` | ~178M (~669 MB) | Apache 2.0 | 0.7733 | 0.7912 | 0.7733 | 0.7705 | ~22.4 ms | Alto (huella de memoria pesada) |

> *Nota: Valores derivados de la ejecución reproducible registrada en `results/metricas_comparativas.csv` ejecutada sobre GPU mediante el cuaderno en `notebooks/`.*

### Análisis de Trade-offs (Eficiencia frente a Precisión)
* **DistilBERT SST-2**: Demostró ser el modelo superior en ambos frentes. Gracias a la técnica de destilación de conocimiento (*knowledge distillation*), retiene el 97% de la capacidad de BERT reduciendo su tamaño a casi una tercera parte, logrando el F1-Score más alto (**0.8461**) y una latencia de inferencia ultra veloz por muestra.
* **Twitter RoBERTa**: Ofrece solidez ante lenguaje informal y abreviaciones, pero su latencia es casi un 75% mayor que la de DistilBERT sin ofrecer una mejora en exactitud sobre texto estructurado.
* **Multilingual BERT**: Al mantener cabezales de 5 clases proyectados a clasificación por estrellas y soportar múltiples idiomas simultáneamente, incurre en mayor degradación métrica (F1 de 0.7705) y latencia elevada, descartándolo para un sistema focalizado en texto en inglés.

### Decisiones de Ajuste Mínimo (*Minimal Adaptation*)
1. **Normalización de Salidas**: Se implementó una lógica de unificación de etiquetas que mapea esquemas heterogéneos (`LABEL_0/1/2`, escalas de 1 a 5 estrellas y etiquetas binarias) al estándar del proyecto (`POS`, `NEG`, `NEU`).
2. **Truncamiento de Secuencias**: Se acotaron las entradas a 256 tokens (`truncation=True, max_length=256`). Dado que las reseñas de IMDB pueden superar los 1,000 tokens, este ajuste reduce el consumo cuadrático de memoria del mecanismo de autoatención sin sacrificar el contexto crítico del inicio del texto.
3. **Aceleración por Hardware**: Asignación explícita de `device=0` en el pipeline de Transformers para canalizar los tensores directamente a los núcleos CUDA de la GPU.

## 5. Justificación del Modelo Recomendado
Se selecciona formalmente **`distilbert-base-uncased-finetuned-sst-2-english`** como la arquitectura recomendada para el pipeline de producción:
* **Cumplimiento de Umbral**: Supera el estándar de calidad fijado (F1-score ≥ 0.80), alcanzando **0.8461**.
* **Eficiencia Operativa**: Al registrar la menor latencia (~8.5 ms/muestra) y el tamaño más liviano en disco (~268 MB), minimiza los costos de cómputo en infraestructuras de microservicios o entornos contenerizados.
* **Seguridad Jurídica**: Licencia Apache 2.0 que autoriza su explotación libre en entornos comerciales y académicos.

### Umbrales y Riesgos Operativos
* **Umbral Mínimo**: $F_1 \ge 0.80$. Caídas de rendimiento por debajo de este indicador en monitoreo continuo detonarán ciclos de ajuste fino supervisado (*fine-tuning* de dominio).
* **Riesgo de Sarcasmo y Negaciones Dobles**: Modelos basados en transformadores compactos pueden interpretar erróneamente contrastes como *"Not bad at all"* o ironías complejas. Se sugiere enrutamiento a validación humana para inferencias con puntaje de confianza (*confidence score*) $< 0.65$.
* **Deriva Textual**: Reseñas con jergas modernas no presentes en el corpus de preentrenamiento pueden erosionar la precisión a lo largo del tiempo.

## 6. Entorno de Ejecución
* **Plataforma**: Google Colab (Acelerador: GPU CUDA).
* **Entorno**: Python 3.10+
* **Instalación de Dependencias**:
```bash
pip install -r requirements.txt

## 7. Pasos de Ejecución

### Opción A: Ejecución interactiva en Google Colab (Recomendada)
1. Navega a la carpeta `notebooks/` y abre el archivo `actividad_4_prototipado.ipynb`.
2. Haz clic en la insignia o enlace para abrirlo directamente en Google Colab con aceleración por GPU.
3. Ejecuta las celdas de forma secuencial para inspeccionar la descarga de datos, inferencia y tabla final de métricas.

### Opción B: Ejecución local por terminal
```bash
git clone <URL_DE_TU_REPOSITORIO>
cd analisis-sentimientos-nlp
pip install -r requirements.txt
python src/inference_pipeline.py
