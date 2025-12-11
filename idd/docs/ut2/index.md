---
title: "Unidad 2: Preprocesamiento y Transformación"
---

# Unidad 2: Preprocesamiento y Transformación de Datos

Esta unidad se enfoca en las técnicas esenciales de preprocesamiento de datos, un paso fundamental antes del modelado. Aprenderás a manejar datos faltantes, normalizar variables y detectar sesgos en tus datasets.

## 📚 Contenido de la Unidad

### [Práctica 4: Missing Data Detective](04-missing-data.md)
**Identificación y tratamiento de datos faltantes**

Aprende a detectar y manejar valores faltantes de manera profesional:
- Identificación de patrones de datos faltantes
- Técnicas de imputación (media, mediana, moda)
- Imputación avanzada con algoritmos
- Evaluación del impacto de datos faltantes
- Estrategias de eliminación vs imputación

### [Práctica 5: Feature Scaling & Anti-Leakage Pipeline](05-feature-scaling.md)
**Normalización de variables y prevención de fugas de datos**

Domina las técnicas de escalado de características y evita errores comunes:
- Feature Scaling: StandardScaler, MinMaxScaler, RobustScaler
- Construcción de pipelines robustos
- Prevención de data leakage
- Train-test split correcto
- Cross-validation sin fugas de información
- Dataset: Ames Housing

### [Práctica 6: Detectar y Corregir Sesgo con Fairlearn](06-detect-correct-bias.md)
**Análisis de equidad y corrección de sesgos algorítmicos**

Aprende a identificar y mitigar sesgos en modelos de Machine Learning:
- Detección de sesgos en datos y modelos
- Métricas de equidad (disparate impact, demographic parity)
- Uso de Fairlearn para análisis de fairness
- Técnicas de mitigación de sesgo
- Evaluación de trade-offs entre precisión y equidad
- Implementación de modelos justos

## 🎯 Objetivos de Aprendizaje

Al completar esta unidad, serás capaz de:

- ✅ Identificar y tratar datos faltantes de manera efectiva
- ✅ Aplicar técnicas de normalización y escalado apropiadas
- ✅ Construir pipelines de preprocesamiento robustos
- ✅ Prevenir data leakage en proyectos de ML
- ✅ Detectar sesgos en datos y modelos
- ✅ Implementar técnicas de fairness en ML
- ✅ Evaluar modelos desde perspectivas de equidad

## 🛠️ Herramientas y Tecnologías

- **Pandas**: Manipulación de datos
- **Scikit-learn**: Escalado y pipelines
- **Fairlearn**: Análisis y mitigación de sesgos
- **NumPy**: Operaciones numéricas
- **Matplotlib & Seaborn**: Visualización
- **Ames Housing Dataset**: Dataset principal de práctica

## ⚠️ Conceptos Clave

- **Data Leakage**: Filtración de información del conjunto de test al entrenamiento
- **Feature Scaling**: Normalización de variables para mejorar el rendimiento de modelos
- **Fairness**: Equidad en predicciones para diferentes grupos demográficos
- **Imputation**: Técnicas para rellenar valores faltantes
