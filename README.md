# 🧬 ML_Cancer — Clasificación de Tipos de Cáncer mediante Expresión Génica

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-✓-green)](https://xgboost.readthedocs.io/)
[![UMAP](https://img.shields.io/badge/UMAP-✓-purple)](https://umap-learn.readthedocs.io/)

> Proyecto de Machine Learning para clasificar automáticamente muestras tumorales en 5 tipos de cáncer a partir de datos de expresión génica (RNA-seq), con una accuracy superior al 97%.

---

## 📋 Índice

- [Descripción del proyecto](#-descripción-del-proyecto)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Dataset](#-dataset)
- [Pipeline de modelado](#-pipeline-de-modelado)
- [Resultados](#-resultados)
- [Instalación y uso](#-instalación-y-uso)
- [Conclusiones](#-conclusiones)

---

## 🔬 Descripción del proyecto

El diagnóstico oncológico preciso es fundamental para mejorar las tasas de supervivencia. Este proyecto explora el uso del Machine Learning para clasificar muestras tumorales en distintos tipos de cáncer a partir de sus **perfiles de expresión génica**, es decir, midiendo qué genes están activos y en qué grado en cada muestra.

Los datos provienen del proyecto **TCGA (The Cancer Genome Atlas)** y contienen miles de variables genómicas por muestra, lo que convierte este problema en un reto de alta dimensionalidad ideal para técnicas de reducción como PCA y UMAP.

**Tipos de tumor clasificados:**

| Código | Tipo de cáncer |
|--------|---------------|
| BRCA | Cáncer de mama (Breast Carcinoma) |
| KIRC | Carcinoma de células renales |
| COAD | Adenocarcinoma de colon |
| LUAD | Adenocarcinoma de pulmón |
| PRAD | Adenocarcinoma de próstata |

---

## 📁 Estructura del repositorio

```
ML_Cancer/
│
├── src/
│   ├── data/
│   │   ├── train.csv           # Datos de entrenamiento (expresión génica + etiquetas)
│   │   └── test.csv            # Datos de test
│   │
│   ├── notebooks/
│   │   ├── 01_EDA.ipynb        # Análisis exploratorio de datos
│   │   ├── 02_preprocessing.ipynb  # Limpieza y preprocesamiento
│   │   └── 03_modeling.ipynb   # Iteraciones y comparativa de modelos
│   │
│   ├── utils/
│   │   └── helpers.py          # Funciones auxiliares (preprocesado, visualización, etc.)
│   │
│   ├── model/
│   │   ├── production/
│   │   │   └── logistic_regression_pca.pkl  # ✅ Modelo final en producción
│   │   ├── svm_model.pkl
│   │   ├── random_forest_model.pkl
│   │   └── xgboost_model.pkl
│   │
│   └── memoria.ipynb           # Memoria limpia del proyecto
│
└── README.md
```

---

## 📊 Dataset

- **Fuente:** [TCGA — The Cancer Genome Atlas](https://www.cancer.gov/tcga)
- **Formato:** Matriz de expresión génica (muestras × genes) + archivo de etiquetas
- **Tamaño:** Miles de variables génicas por muestra
- **Clases:** 5 tipos de tumor (ver tabla superior)
- **Desbalance:** BRCA es la clase más representada → se usa estratificación en el split train/test

---

## ⚙️ Pipeline de modelado

```
Datos brutos (expresión génica)
        ↓
LabelEncoder (codificación de etiquetas)
        ↓
Train/Test Split estratificado (80/20)
        ↓
StandardScaler (estandarización)
        ↓
PCA (reducción al 95% de varianza explicada)
        ↓
Clasificador (Regresión Logística ✅)
        ↓
Predicción del tipo de tumor
```

**Modelos evaluados:**

| Modelo | CV Accuracy | Seleccionado |
|--------|-------------|:---:|
| Logistic Regression | ~0.98 ± bajo | ✅ |
| SVM (RBF) | ~0.97 ± bajo | |
| Random Forest | ~0.97 ± bajo | |
| XGBoost | ~0.97 ± bajo | |
| AdaBoost | ~0.96 ± bajo | |

> La validación cruzada estratificada de 5 folds se realizó sobre los datos de entrenamiento reducidos con PCA.

---

## 📈 Resultados

- **Accuracy en test:** > 97%
- **Precision, Recall y F1-score** altos en todas las clases
- Los errores de clasificación se producen entre tipos de tumor biológicamente similares
- La visualización UMAP confirma clusters compactos y bien separados para cada tipo

**Extensión no supervisada:** Se aplicó KMeans sobre las muestras BRCA (tras PCA de 50 componentes) para identificar posibles subtipos moleculares (Luminal A, Luminal B, HER2-enriched, Basal-like), coherentes con la literatura biomédica.

---

## 🚀 Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/fgoiri/ML_Cancer.git
cd ML_Cancer
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Librerías principales:**
- `scikit-learn`
- `xgboost`
- `umap-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `joblib`

### 3. Ejecutar los notebooks

Abre los notebooks en orden dentro de `src/notebooks/`:
1. `01_EDA.ipynb` — Exploración del dataset
2. `02_preprocessing.ipynb` — Preprocesamiento y PCA
3. `03_modeling.ipynb` — Entrenamiento y comparativa de modelos

O consulta directamente la **memoria del proyecto** en `src/memoria.ipynb` para un resumen completo del proceso.

### 4. Usar el modelo en producción

```python
import joblib
import pandas as pd

# Cargar pipeline completo
model = joblib.load('src/model/production/logistic_regression_pca.pkl')

# Predecir sobre nuevas muestras
# X_new: DataFrame con los mismos genes que el dataset original
predictions = model.predict(X_new)
print(predictions)  # ['BRCA', 'LUAD', 'KIRC', ...]
```

---

## 🧠 Conclusiones

Este proyecto demuestra que es posible clasificar el tipo de cáncer con más de un **97% de accuracy** a partir únicamente de datos de expresión génica, utilizando un pipeline relativamente sencillo:

- ✅ Sin data leakage (scaler y PCA ajustados solo en train)
- ✅ Validación cruzada estratificada para estimaciones robustas
- ✅ Reducción de dimensionalidad efectiva (de miles de genes a componentes PCA)
- ✅ Extensión de clustering para identificación de subtipos moleculares en BRCA

**Futuras mejoras:** optimización de hiperparámetros (Optuna/GridSearch), análisis de interpretabilidad con SHAP, validación externa en cohortes independientes, y exploración de redes neuronales profundas.

---

*Proyecto desarrollado en el marco del Bootcamp de Data Science & Machine Learning.*
*Herramientas: Python · scikit-learn · XGBoost · UMAP · pandas · matplotlib · seaborn*v
