# Dog vs Cat Classification

Este proyecto entrena una CNN con TensorFlow/Keras para clasificar imágenes de gatos y perros a partir de un dataset de Kaggle.

## Cómo se ha hecho

Descarga del dataset con kagglehub.

Exploración inicial para comprobar balanceo de clases y visualizar ejemplos.

Filtrado de imágenes corruptas o con problemas de formato (rotas, truncadas o con número de canales no válido) usando PIL.

Preprocesado: redimensionado a 180x180 píxeles, normalización de valores [0,1] y conversión a RGB.

División estratificada en entrenamiento y test.

Definición de un modelo CNN secuencial con varias capas convolucionales, pooling, batch normalization y dropout.

Entrenamiento con callbacks (EarlyStopping, ReduceLROnPlateau).

Evaluación mediante accuracy, loss, matriz de confusión y visualización de predicciones junto a sus probabilidades.

## Problemas encontrados y soluciones

Imágenes corruptas o truncadas: filtradas con PIL.Image.verify y ImageFile.LOAD_TRUNCATED_IMAGES = True.

Formatos inesperados (BMP, PNG, 2 canales, animadas): se unificaron usando PIL dentro de tf.py_function, forzando conversión a RGB.

Errores de TensorFlow con decode_jpeg y decode_image: resueltos descartando imágenes no compatibles y usando un preprocesado estable con PIL.

## Resultados

El modelo alcanza alrededor de 80–86% de accuracy en el conjunto de validación.

Se han generado curvas de entrenamiento, matriz de confusión e imágenes con sus predicciones y probabilidades asociadas.
