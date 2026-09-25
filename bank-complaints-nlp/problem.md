# Problema: clasificación automática de quejas de clientes en JPMorgan Chase

> Caso hipotético con fines educativos. Los datos son quejas reales y públicas contra JPMorgan Chase, tomadas de la
> base de quejas del regulador de EE. UU. (CFPB). Las cifras de operación del caso son ilustrativas.

## 1. Contexto

**JPMorgan Chase** recibe decenas de miles de quejas al año, en texto libre, por su web, su app, el correo y el
portal del regulador (CFPB). Un equipo de analistas lee cada queja y la envía a mano al área que debe resolverla.

## 2. El problema

- Cerca del **30 %** de las quejas llega primero al área equivocada, y cada reenvío cuesta días.
- El CFPB espera que el banco **responda en 15 días** y cierre la mayoría de casos en 60 días.
- La categoría que el cliente marca en el formulario no es confiable, así que **no hay etiquetas buenas** para
  entrenar un modelo.

**Pregunta:** *¿podemos leer cada queja automáticamente y enviarla al equipo correcto?*

## 3. Datos

`complaints.json` (~78 mil registros con el esquema del CFPB, de `SparkNLP-Project-Bank-Complaints`). Es un dataset
público de Kaggle:
[`sebastianramirezesc/complaints-json`](https://www.kaggle.com/datasets/sebastianramirezesc/complaints-json)
(se agrega con *Add Input*).

| Campo | Uso |
| --- | --- |
| `complaint_what_happened` | Texto de la queja (entrada principal). Se descartan las vacías; los datos personales vienen enmascarados como `XXXX` |
| `product`, `issue` | Producto y problema que marcó el cliente (ruidosos). Se convierten en las 5 categorías y sirven de referencia para la comparación |

## 4. Categorías de enrutamiento

| Id | Categoría | Equipo |
| --- | --- | --- |
| 0 | Hipotecas y préstamos | Préstamos |
| 1 | Cuentas y transacciones | Depósitos |
| 2 | Reportes de crédito | Burós de crédito |
| 3 | Tarjetas de crédito | Tarjetas |
| 4 | Fraude y disputas | Fraude |

## 5. Tareas (un notebook de Kaggle)

Configuración del notebook: **Accelerator: GPU T4** (o P100) e **Internet: On**.

1. **Preparación:** Python en Kaggle con pandas, spaCy (`en_core_web_sm`), scikit-learn y transformers.
2. **EDA y limpieza:** explorar texto y metadatos (consentimiento, canal, fecha, producto, problema) para entender
   por qué el 73 % de las quejas no tiene texto y qué sesgo introduce eso. Luego pasar a minúsculas y quitar
   puntuación, números y `XXXX`.
3. **Preprocesamiento (spaCy):** tokenizar → quitar *stop words* → lematizar → etiquetar POS → conservar solo
   sustantivos.
4. **Vectorización:** separar entrenamiento y prueba **antes** de ajustar nada (sin fuga de datos). Usar
   `CountVectorizer` de scikit-learn (palabras + bigramas, `max_df=0.5`) y `TfidfTransformer`.
5. **Temas:** `LatentDirichletAllocation` con k = 5, α = 0,05 y aprendizaje *batch*. Entrenar 10 semillas y quedarse
   con la de menor perplejidad. Leer las palabras clave, nombrar cada tema con una categoría, visualizar la separación
   de los temas con un mapa t-SNE (distancia de Hellinger) y usar el tema dominante de cada queja como su etiqueta
   de entrenamiento.
6. **Clasificación:** entrenar un `Pipeline` (TF-IDF + Regresión Logística) con una división 70/30 y reportar
   accuracy, F1 ponderado y F1 macro.
7. **Comparar dos enfoques** sobre la misma muestra de prueba (~500 quejas), evaluados contra la categoría de
   referencia:

   | Enfoque | Modelo | ¿Necesita etiquetas humanas? |
   | --- | --- | --- |
   | A. spaCy + LDA + Regresión Logística | Tu modelo del paso 6 | No (usa los temas de LDA) |
   | B. Clasificador zero-shot NLI | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` de Hugging Face, con descripciones de cada categoría | No |

   Reportar accuracy, F1 ponderado, F1 macro, matriz de confusión y tiempo por cada 100 quejas para cada uno.

8. **Conclusión:** ¿qué enfoque debería usar Chase y por qué? Considerar calidad, velocidad, costo y la necesidad de
   etiquetas.

## 6. Criterios de éxito

- El notebook corre de principio a fin en una sesión gratuita de Kaggle con GPU (T4 o P100).
- Los temas de LDA se pueden leer y corresponden a las 5 categorías.
- El enfoque A alcanza un accuracy de **0,70** o más **contra la categoría de referencia**, no contra sus propios
  temas de LDA, porque eso sería una evaluación circular.
- La tabla comparativa y la conclusión se apoyan en los números.
