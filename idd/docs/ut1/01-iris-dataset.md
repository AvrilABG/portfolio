---
title: "Práctico 01: Iris dataset"
date: 2025-01-01
---

En este práctico veremos un analisis al famoso iris dataset.

## Preparación del entorno
Primero empezamos esta prática instalando las librerías necesarias en Python.

```PowerShell
pip install pandas seaborn matplotlib sklearn
```

Después verificamos las librerías y la versión de python

```python
# Importación de librerías
import seaborn as sns
import sys
import pandas as pd
import seaborn as sns
import matplotlib

# Verificación de la versión de Python
print(sys.version)

# Verificación de las versiones de las librerías
print(pd.__version__, sns.__version__, matplotlib.__version__)
```

## Investigación del dataset
Investigamos el dataset, empezando a realizar preguntas mientras su exploración.  
Ejemplos de preguntas ordenadas de mayor o menor:
1. ¿Que correlaciones hay entre los tamaños del sepalo y del petalo?
2. ¿Cuál es el promedio de largo del petalo por especie?
3. ¿Cuál es el promedio del largo del sepalo por especie?

Y ahora, cargaremos el dataset a través de seaborn

```python
df = sns.load_dataset('iris')
df.head()
```
Aunque existan otras maneras para cargar este dataset, como desde Kaggle Hub o de scikit-learn.

## Data dictionary
Acá se mostrará un data dictionary del dataset
| Nombre        | Tipo      | Descripción               | Unidad
|---------------|-----------|---------------------------|-------|
| `sepal_length`| `float64` | Ancho del sepalo del iris | cm   
| `sepal_width` | `float64` | Largo del sepalo del iris | cm
| `petal_length`| `float64` | Largo del petalo del iris | cm
| `petal_width` | `float64` | Ancho del petalo del iris | cm
| `species`     | `category`| Clase de Iris: Iris Setosa, Iris Versicolour, o Iris Virginica  |             
