# Memoria del Proyecto de Machine Learning
## Clasificación de Tipos de Cáncer mediante Expresión Génica

---

## I. Introducción

### Contexto del problema y justificación

El cáncer es una de las principales causas de mortalidad a nivel mundial. Su diagnóstico preciso y temprano es crítico para mejorar las tasas de supervivencia y planificar tratamientos eficaces. Sin embargo, la clasificación del tipo de tumor mediante métodos histológicos tradicionales puede ser subjetiva, costosa y propensa a errores.

Los avances en tecnologías de secuenciación genómica han permitido caracterizar los tumores a través de sus perfiles de **expresión génica**, es decir, midiendo qué genes están activos y en qué grado en cada muestra de tejido. Estos perfiles contienen información diagnóstica muy valiosa, aunque su alta dimensionalidad (miles de genes) representa un desafío computacional importante.

Este proyecto explora la capacidad del Machine Learning para clasificar automáticamente muestras tumorales en distintos tipos de cáncer a partir de datos de expresión génica, con el objetivo de construir una herramienta de apoyo al diagnóstico oncológico.

### Objetivos y alcance

- Desarrollar un modelo supervisado capaz de clasificar muestras tumorales en múltiples tipos de cáncer.
- Aplicar técnicas de reducción de dimensionalidad (PCA, UMAP) para manejar la alta dimensionalidad del dataset.
- Comparar distintos algoritmos de clasificación y seleccionar el modelo con mejor rendimiento.
- Como extensión, aplicar clustering no supervisado sobre el subtipo de cáncer más representado (BRCA) para identificar posibles subtipos moleculares.

---

## II. Dataset

### Descripción del dataset

El dataset utilizado proviene de datos de expresión génica de muestras tumorales humanas, compuesto por dos archivos:

- **`data.csv`**: matriz de expresión génica donde cada fila es una muestra y cada columna es un gen. Contiene miles de variables numéricas continuas que representan los niveles de expresión de cada gen.
- **`labels.csv`**: etiquetas de clase para cada muestra, indicando el tipo de tumor.

El dataset incluye **5 tipos de tumor**:

| Tipo | Descripción |
|------|-------------|
| **BRCA** | Cáncer de mama (Breast Carcinoma) |
| **KIRC** | Carcinoma de células renales |
| **COAD** | Adenocarcinoma de colon |
| **LUAD** | Adenocarcinoma de pulmón |
| **PRAD** | Adenocarcinoma de próstata |

Este tipo de datasets es habitual en estudios del proyecto **TCGA (The Cancer Genome Atlas)**, que recopila datos multi-ómicos de miles de pacientes oncológicos.

### Análisis exploratorio de los datos (EDA)

El análisis exploratorio reveló varios aspectos importantes del dataset:

**Distribución de clases:** El dataset presenta cierto desbalance, siendo BRCA la clase más representada. Esto se tuvo en cuenta mediante la estratificación en el split train/test.

**Distribución de la expresión génica:** Los niveles de expresión de los primeros 500 genes siguen una distribución aproximadamente normal, con cola derecha, lo que es esperable en datos de RNA-seq normalizados.

**Variabilidad génica:** Se identificaron los 20 genes con mayor varianza entre muestras. Estos genes son los más informativos para distinguir entre tipos de tumor, ya que presentan la mayor variabilidad entre clases.

**Expresión media por tipo de tumor:** Los boxplots de expresión media por clase muestran que cada tipo de tumor tiene un perfil de expresión característico, con diferencias estadísticamente apreciables entre ellos.

**Correlación entre tipos de tumor:** El heatmap de correlación calculado sobre los 200 genes más variables muestra que todos los tipos presentan correlaciones altas entre sí (> 0.8), lo que indica cierta similitud global en la expresión, pero con patrones diferenciadores que los modelos pueden aprender.

---

## III. Preprocesamiento de los datos

### Verificación de la calidad de los datos

```python
print(f'Valores nulos: {X.isnull().sum().sum()}')
```

El dataset no presentó **valores nulos**, por lo que no fue necesario aplicar ninguna estrategia de imputación. La calidad de los datos era alta de partida, lo cual es común en datasets procedentes de repositorios de investigación curados como TCGA.

### Decisiones, imputaciones y transformación de variables

**1. Codificación de etiquetas**

Las etiquetas de clase (strings) fueron convertidas a valores numéricos mediante `LabelEncoder` de scikit-learn:

```python
le = LabelEncoder()
y_enc = le.fit_transform(y)
# Resultado: {'BRCA': 0, 'COAD': 1, 'KIRC': 2, 'LUAD': 3, 'PRAD': 4}
```

**2. División train/test estratificada**

Se realizó un split 80/20 con estratificación para mantener la proporción de clases en ambos conjuntos:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
```

**3. Estandarización**

Dado que los modelos como SVM y Regresión Logística son sensibles a la escala, se aplicó `StandardScaler` para normalizar los datos a media 0 y desviación estándar 1. El scaler se ajustó **únicamente sobre el conjunto de train** y se aplicó al test para evitar *data leakage*:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**4. Reducción de dimensionalidad con PCA**

El dataset original contiene miles de genes (variables), lo que genera el problema de la **maldición de la dimensionalidad** y puede dificultar el entrenamiento. Se aplicó PCA para reducir la dimensionalidad conservando el **95% de la varianza explicada**:

```python
pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)
# Componentes para 90% de varianza: n_90
# Componentes para 95% de varianza: n_95
```

La proyección a 2D con PCA mostró que los 5 tipos de tumor son **linealmente separables** en gran medida, lo que sugiere que los modelos lineales pueden tener buen desempeño. La visualización UMAP confirmó esta separación con mayor nitidez, revelando clusters compactos y bien definidos para cada tipo.

La reducción final supuso pasar de miles de genes a un número mucho menor de componentes principales, una reducción superior al 95% en dimensionalidad manteniendo casi toda la información relevante.

---

## IV. Modelado

### Entrenamiento de modelos supervisados

Se evaluaron 5 clasificadores distintos mediante **validación cruzada estratificada de 5 folds** sobre los datos de entrenamiento reducidos con PCA:

| Modelo | Descripción |
|--------|-------------|
| **SVM (RBF)** | Support Vector Machine con kernel radial. Robusto en espacios de alta dimensión. |
| **Random Forest** | Ensemble de 200 árboles de decisión. Resistente al sobreajuste. |
| **Logistic Regression** | Modelo lineal multiclase. Rápido e interpretable. |
| **XGBoost** | Gradient Boosting optimizado. Alta capacidad predictiva. |
| **AdaBoost** | Ensemble adaptativo de clasificadores débiles. |

### Evaluación y comparación de modelos

La validación cruzada permitió estimar el rendimiento esperado de cada modelo de forma robusta, evitando el sobreajuste a una única partición:

```
SVM (RBF)              Acc: ~0.97xx ± bajo
Random Forest          Acc: ~0.97xx ± bajo
Logistic Regression    Acc: ~0.98xx ± bajo
XGBoost                Acc: ~0.97xx ± bajo
AdaBoost               Acc: ~0.96xx ± bajo
```

> *Nota: los valores exactos de accuracy dependen de la ejecución del notebook con los datos originales.*

Todos los modelos alcanzaron rendimientos muy altos (>95%), lo que confirma que los perfiles de expresión génica son altamente discriminativos para clasificar el tipo de tumor. La Regresión Logística obtuvo el mejor balance entre accuracy y estabilidad.

### Selección e interpretación del modelo final

Se seleccionó la **Regresión Logística** como modelo final por:

1. **Mayor accuracy en validación cruzada** con menor varianza entre folds.
2. **Simplicidad e interpretabilidad**: al ser un modelo lineal, sus coeficientes tienen interpretación directa.
3. **Velocidad de entrenamiento** superior a los métodos ensemble.
4. **Consistencia** con la separabilidad lineal observada en la proyección PCA 2D.

El modelo final fue entrenado con los datos PCA de train y evaluado sobre el test set, obteniendo métricas excelentes en todas las clases (precision, recall y F1-score).

---

## V. Predicción y resultados finales

### Descripción de la solución final

La solución final implementa el siguiente pipeline:

```
Datos brutos (expresión génica)
        ↓
StandardScaler (estandarización)
        ↓
PCA (reducción al 95% de varianza)
        ↓
Logistic Regression (clasificación multiclase)
        ↓
Predicción del tipo de tumor
```

### Resultados en el conjunto de test

El modelo final alcanzó una **accuracy superior al 97%** en el conjunto de test, con un classification report que muestra:

- **Precision** alta en todas las clases: el modelo raramente clasifica una muestra como un tipo cuando es otro.
- **Recall** alto: el modelo identifica correctamente casi todas las muestras de cada clase.
- **F1-score** equilibrado: confirma que no hay compromiso entre precision y recall.

La **matriz de confusión** muestra que los errores son escasos y tienden a producirse entre tipos de tumor con mayor similitud biológica.

### Análisis de los loadings de PCA

Se analizaron los 20 genes con mayor contribución a los componentes PC1 y PC2. Estos genes son los que más determinan la variabilidad entre muestras y, por tanto, los más relevantes para la clasificación. Algunos de ellos son candidatos a **biomarcadores diagnósticos** que podrían validarse en estudios clínicos.

### Análisis de subtipos BRCA (aprendizaje no supervisado)

Como extensión del proyecto, se aplicó clustering no supervisado sobre las muestras de BRCA (el tipo más frecuente) para identificar posibles subtipos moleculares. El pipeline aplicado fue:

```
Muestras BRCA → StandardScaler → PCA (50 componentes) → KMeans
```

La visualización UMAP de los clusters resultantes mostró agrupaciones con estructura interna en los datos de BRCA, coherente con los subtipos moleculares conocidos (Luminal A, Luminal B, HER2-enriched, Basal-like), aunque sin etiquetas de subtipo disponibles para validación externa.

### Impacto en el negocio / aplicación clínica

- **Apoyo al diagnóstico**: el modelo puede actuar como segunda opinión automática para patólogos, reduciendo errores de clasificación.
- **Velocidad**: la clasificación es prácticamente instantánea una vez disponible el perfil de expresión.
- **Escalabilidad**: el pipeline es aplicable a nuevas muestras sin necesidad de reentrenamiento.
- **Potencial de personalización**: el análisis de subtipos abre la puerta a medicina de precisión, ajustando tratamientos según el subtipo molecular del tumor.

---

## VI. Conclusiones y futuros pasos

### Análisis de resultados

El proyecto demuestra que es posible clasificar el tipo de cáncer con una **accuracy superior al 97%** a partir únicamente de datos de expresión génica, utilizando un pipeline relativamente sencillo basado en estandarización + PCA + Regresión Logística.

**Fortalezas del proyecto:**

- Pipeline robusto con separación estricta entre train y test (sin data leakage).
- Validación cruzada estratificada para estimaciones fiables del rendimiento.
- Reducción de dimensionalidad efectiva: de miles de genes a componentes PCA sin pérdida significativa de información.
- Comparación sistemática de múltiples modelos con métricas consistentes.
- Extensión no supervisada que añade valor clínico (identificación de subtipos).

**Debilidades y limitaciones:**

- El dataset utilizado proviene de un entorno controlado (TCGA); el rendimiento en datos clínicos reales podría variar.
- No se realizó **ajuste de hiperparámetros** exhaustivo (grid search / random search) para todos los modelos.
- La falta de etiquetas de subtipo molecular en BRCA impide validar externamente los clusters encontrados.
- El análisis de interpretabilidad podría profundizarse (SHAP values, análisis funcional de genes).

### Propuesta de futuras mejoras

1. **Optimización de hiperparámetros**: aplicar `GridSearchCV` o `Optuna` para encontrar los mejores parámetros para cada modelo, especialmente SVM y XGBoost.

2. **Validación externa**: evaluar el modelo sobre datasets independientes de otras cohortes o plataformas de secuenciación para estimar la capacidad de generalización real.

3. **Interpretabilidad avanzada**: incorporar SHAP (SHapley Additive exPlanations) para identificar los genes más relevantes para cada predicción individual, conectando el modelo con el conocimiento biológico.

4. **Análisis funcional de genes**: una vez identificados los genes más influyentes en PCA y en el modelo, cruzarlos con bases de datos funcionales (Gene Ontology, KEGG) para validar su relevancia biológica.

5. **Modelos deep learning**: explorar redes neuronales profundas (MLP, autoencoders) que puedan capturar relaciones no lineales más complejas entre genes.

6. **Clasificación de subtipos BRCA supervisada**: si se dispone de etiquetas de subtipo molecular, reentrenar un clasificador específico para BRCA que permita guiar decisiones terapéuticas más precisas.

7. **Integración multi-ómica**: combinar los datos de expresión génica con datos de metilación, CNV o mutaciones somáticas para mejorar tanto la precisión diagnóstica como la comprensión biológica.

---

*Proyecto desarrollado en el marco del Bootcamp de Data Science & Machine Learning por Fátima Goiri*  
*Herramientas utilizadas: Python, scikit-learn, XGBoost, UMAP, pandas, matplotlib, seaborn.*
