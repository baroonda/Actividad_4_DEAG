# Benchmark y Selección de Modelos Preentrenados para Clasificación de Sentimientos

## 1. Descripción del Caso de Uso
* **Contexto de Negocio**: Implementación de un sistema automatizado para clasificar opiniones de comensales/clientes recopiladas de canales digitales (Google Reviews, encuestas de satisfacción) en tres categorías: Positivo (`POS`), Neutro (`NEU`) y Negativo (`NEG`).
* **Objetivo de IA**: Reducir el tiempo de clasificación manual mediante un modelo de lenguaje preentrenado optimizado para español, priorizando la detección de críticas negativas para atención inmediata.
* **Dominio**: Procesamiento de Lenguaje Natural (NLP).

## 2. Ecosistema Hugging Face y Criterios de Selección
* **Componentes del Ecosistema Utilizados**:
  * **Hugging Face Models**: Extracción de transformadores preentrenados especializados en análisis de opiniones.
  * **Hugging Face Datasets**: Estandarización de flujos de ingesta de datos.
  * **Hugging Face Hub / Spaces**: Consulta de fichas técnicas (*Model Cards*), licencias y demos públicas de validación.
* **Criterios Técnicos y Prácticos**:
  1. *Cobertura del Idioma y Jerga Local*: Alta familiaridad con el español informal y modismos.
  2. *Compatibilidad de Licencias*: Licencias permisivas (MIT, Apache 2.0) aptas para investigación o uso comercial.
  3. *Balance Latencia-Consumo*: Huella de memoria compatible con inferencia en hardware accesible (GPU T4 o CPU estándar).

## 3. Dataset y Soporte Reproducible
* **Origen**: Conjunto representativo derivado de opiniones de clientes en plataformas de reseñas y encuestas de servicio.
* **Estructura de Datos**:
  * `text` (*string*): Contenido de la reseña o comentario dejado por el usuario.
  * `label` (*string*): Clasificación de referencia (`POS`, `NEU`, `NEG`).
* **Reproducibilidad**: Disponible en `data/reviews_dataset.csv`. Los scripts de normalización e ingesta están parametrizados en `src/inference_pipeline.py`.

## 4. Reporte Técnico Comparativo

### Tabla Comparativa de Modelos
| Modelo | Tarea | Tamaño / Parámetros | Licencia | Accuracy | Precision | Recall | F1-Score | Latencia (GPU T4) | Recursos |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RoBERTuito** (`pysentimiento/robertuito-sentiment-analysis`) | `text-classification` | ~100M (~440 MB) | MIT | **0.90** | **0.91** | **0.90** | **0.90** | **14.2 ms** | Bajo (GPU/CPU ligera) |
| **Multilingual BERT** (`nlptown/bert-base-multilingual-...`) | `text-classification` | ~178M (~700 MB) | Apache 2.0 | 0.70 | 0.75 | 0.70 | 0.71 | 24.8 ms | Medio |
| **Twitter XLM-RoBERTa** (`cardiffnlp/twitter-xlm-roberta-...`) | `text-classification` | ~278M (~1.1 GB) | CC-BY-NC | 0.80 | 0.82 | 0.80 | 0.80 | 33.5 ms | Alto |

### Análisis de Trade-offs (Eficiencia vs. Precisión)
* **RoBERTuito** sobresale en rendimiento métrico y velocidad gracias a su tokenización optimizada para lenguaje conversacional en español. Su tamaño compacto minimiza el consumo de VRAM y permite inferencias por debajo de los 15 ms.
* **XLM-RoBERTa** ofrece buena robustez multilingüe, pero su sobrecoste de memoria y latencia (más del doble frente a RoBERTuito) no justifican su uso en un caso estrictamente hispanohablante.

### Decisiones de Ajuste Mínimo (*Minimal Fine-Tuning / Adaptation*)
* **Normalización de Salidas**: Mapeo determinístico de los esquemas de etiquetas originales (`POS/NEU/NEG`, estrellas `1-5` o `positive/negative/neutral`) a una nomenclatura corporativa unificada.
* **Ajuste de Hardware**: Configuración del parámetro `device=0` dentro de `transformers.pipeline` para garantizar aceleración por GPU.
* **Manejo de Truncamiento**: Limitación de secuencias a 128 tokens para evitar saturación de memoria en textos extensos irrelevantes.

## 5. Justificación del Modelo Recomendado
Se selecciona **`pysentimiento/robertuito-sentiment-analysis`** como la opción recomendada:
* Presenta el mayor F1-score ponderado (0.90), superando el umbral de exactitud mínimo requerido del 80%.
* Latencia promedio significativamente inferior (14.2 ms/muestra), lo que reduce drásticamente el costo de cómputo en procesamiento batch o en tiempo real.
* La licencia MIT brinda máxima flexibilidad legal para su adopción.

### Umbrales y Riesgos Operativos
* **Umbral Mínimo**: F1-score ponderado ≥ 0.80. Si una degradación de datos coloca la métrica por debajo de 0.80, se activa una alarma de reentrenamiento.
* **Riesgo de Ambigüedad/Sarcasmo**: Textos con doble sentido o calificaciones neutras tienen mayor probabilidad de falsos positivos/negativos. Mitigación: Enviar predicciones con probabilidad < 0.60 a cola de revisión humana.
* **Deriva de Concepto (Data Drift)**: Nuevos modismos regionales pueden mermar el desempeño. Se sugiere monitoreo mensual.

## 6. Entorno de Ejecución e Instalación
* **Plataforma**: Google Colab (Acelerador: GPU T4).
* **Python**: 3.10+
* **Instalación de Dependencias**:
```bash
pip install -r requirements.txt
