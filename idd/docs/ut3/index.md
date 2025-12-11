# Unidad 3: Feature Engineering Avanzado

Esta unidad cubre técnicas avanzadas de ingeniería de características, esenciales para mejorar el rendimiento de modelos de Machine Learning. Aprenderás a crear, transformar y seleccionar features de manera estratégica.

## 📚 Contenido de la Unidad

### [Práctica 7: Feature Engineering con Pandas](07-feature-engineering.md)
**Creación y transformación de características**

Aprende a crear nuevas variables que mejoren el poder predictivo de tus modelos:
- Creación de features derivadas
- Transformaciones matemáticas (log, sqrt, exponencial)
- Agregaciones y estadísticas por grupo
- Binning y discretización
- Interacciones entre variables
- Feature engineering específico por dominio
- Dataset: Ames Housing

### [Práctica 8: Encoding Avanzado y Target Encoding](08-encoding.md)
**Codificación de variables categóricas**

Domina las técnicas de encoding para variables categóricas:
- One-Hot Encoding y Label Encoding
- Ordinal Encoding
- Target Encoding (Mean Encoding)
- Frequency Encoding
- Binary Encoding
- Hash Encoding
- Manejo de alta cardinalidad
- Prevención de overfitting en encoding

### [Práctica 9: PCA y Feature Selection](09-pca-feature-selection.md)
**Reducción de dimensionalidad y selección de características**

Aprende a reducir la complejidad de tus datos sin perder información relevante:
- Principal Component Analysis (PCA)
- Interpretación de componentes principales
- Variance explained y scree plots
- Feature Selection con métodos filter
- Métodos wrapper (RFE, Forward/Backward Selection)
- Métodos embedded (Lasso, Random Forest importance)
- Técnicas de selección automática
- Trade-offs entre complejidad e interpretabilidad

### [Práctica 10: Temporal Feature Engineering con Pandas](10-temporal-feature.md)
**Ingeniería de características temporales**

Extrae información valiosa de variables de tiempo:
- Extracción de componentes temporales (año, mes, día, hora)
- Features cíclicos (sin/cos transformations)
- Diferencias y deltas temporales
- Rolling windows y agregaciones temporales
- Lag features y features adelantadas
- Tendencias y estacionalidad
- Time-based feature engineering
- Manejo de series temporales

## 🎯 Objetivos de Aprendizaje

Al completar esta unidad, serás capaz de:

- ✅ Crear features nuevas que mejoren el rendimiento de modelos
- ✅ Aplicar transformaciones apropiadas a variables numéricas
- ✅ Codificar variables categóricas de múltiples maneras
- ✅ Reducir dimensionalidad con PCA
- ✅ Seleccionar las características más relevantes
- ✅ Extraer información de variables temporales
- ✅ Construir pipelines completos de feature engineering
- ✅ Evaluar el impacto de nuevas features en el modelo

## 🛠️ Herramientas y Tecnologías

- **Pandas**: Manipulación y feature engineering
- **Scikit-learn**: PCA, feature selection, encoding
- **Category Encoders**: Encoding avanzado
- **NumPy**: Operaciones matemáticas
- **Matplotlib & Seaborn**: Visualización de features
- **Feature Engine**: Herramientas especializadas

## 💡 Conceptos Avanzados

- **Target Encoding**: Codificación basada en la variable objetivo
- **PCA**: Transformación lineal para reducir dimensionalidad
- **Feature Importance**: Medición de la relevancia de variables
- **Curse of Dimensionality**: Problemas con demasiadas features
- **Leakage Prevention**: Evitar filtración de información en encoding
- **Cyclic Features**: Representación de variables cíclicas (tiempo)

## 📊 Datasets Utilizados

- **Ames Housing**: Predicción de precios de viviendas
- **Datasets temporales**: Para feature engineering temporal
- **Datos con alta cardinalidad**: Para técnicas de encoding avanzado
