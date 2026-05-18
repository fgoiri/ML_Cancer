"""
utils/helpers.py
Funciones auxiliares para el proyecto ML_Cancer.
Clasificación de tipos de tumor mediante expresión génica (TCGA PANCAN HiSeq).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# Paleta de colores consistente para los 5 tipos de tumor
TUMOR_COLORS = {
    0: '#4C72B0',  # BRCA
    1: '#DD8452',  # COAD
    2: '#55A868',  # KIRC
    3: '#C44E52',  # LUAD
    4: '#8172B2',  # PRAD
}


# ──────────────────────────────────────────────
# 1. CARGA DE DATOS
# ──────────────────────────────────────────────

def load_data(data_path: str, labels_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga y devuelve features (X) y etiquetas (y) desde los CSV de TCGA.

    Args:
        data_path:   ruta a data.csv (matriz de expresión génica)
        labels_path: ruta a labels.csv (etiquetas de clase)

    Returns:
        X: DataFrame de features
        y: Series con las etiquetas de clase (strings)
    """
    X = pd.read_csv(data_path, index_col=0)
    labels = pd.read_csv(labels_path)
    y = labels['Class'].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y


# ──────────────────────────────────────────────
# 2. VISUALIZACIÓN EDA
# ──────────────────────────────────────────────

def plot_class_distribution(y: pd.Series, save_path: str = None) -> None:
    """
    Gráfico de barras con la distribución de clases (tipos de tumor).

    Args:
        y:         Series con las etiquetas de clase
        save_path: si se indica, guarda la figura en esa ruta
    """
    counts = y.value_counts()
    colors = list(TUMOR_COLORS.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title('Distribución de tipos de tumor', fontsize=13)
    ax.set_ylabel('Número de muestras')
    ax.set_xlabel('Tipo de tumor')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_top_variable_genes(X: pd.DataFrame, n: int = 20, save_path: str = None) -> pd.Series:
    """
    Barplot con los N genes de mayor varianza entre muestras.

    Args:
        X:         DataFrame de features
        n:         número de genes a mostrar (default 20)
        save_path: si se indica, guarda la figura en esa ruta

    Returns:
        Series con los genes más variables (índice=nombre gen, valor=varianza)
    """
    gene_variance = X.var(axis=0)
    top_genes = gene_variance.nlargest(n)

    plt.figure(figsize=(10, 5))
    plt.bar(range(n), top_genes.values, color='steelblue', alpha=0.8, edgecolor='white')
    plt.xticks(range(n), top_genes.index, rotation=45, ha='right', fontsize=8)
    plt.title(f'Top {n} genes con mayor varianza', fontsize=13)
    plt.ylabel('Varianza')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()

    print(f'Gen con mayor varianza: {top_genes.index[0]} ({top_genes.iloc[0]:.2f})')
    print(f'Genes con varianza = 0: {(gene_variance == 0).sum()}')
    return top_genes


def plot_expression_by_tumor(X: pd.DataFrame, y: pd.Series, save_path: str = None) -> None:
    """
    Boxplot de la expresión génica media por tipo de tumor.

    Args:
        X:         DataFrame de features
        y:         Series con las etiquetas de clase
        save_path: si se indica, guarda la figura en esa ruta
    """
    df_plot = pd.DataFrame({
        'expresion_media': X.mean(axis=1),
        'tumor': y.values
    })

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df_plot, x='tumor', y='expresion_media', hue='tumor',
        palette=list(TUMOR_COLORS.values()), legend=False
    )
    plt.title('Expresión génica media por tipo de tumor', fontsize=13)
    plt.xlabel('Tipo de tumor')
    plt.ylabel('Expresión media')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# 3. VISUALIZACIÓN PCA
# ──────────────────────────────────────────────

def plot_pca_variance(pca_full: PCA, save_path: str = None) -> tuple[int, int]:
    """
    Curva de varianza explicada acumulada con marcadores al 90% y 95%.

    Args:
        pca_full:  objeto PCA ya ajustado sobre los datos de train
        save_path: si se indica, guarda la figura en esa ruta

    Returns:
        (n_90, n_95): número de componentes para 90% y 95% de varianza
    """
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_90 = int(np.argmax(cumvar >= 0.90) + 1)
    n_95 = int(np.argmax(cumvar >= 0.95) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(cumvar[:200], color='steelblue', linewidth=2)
    plt.axhline(0.90, color='orange', linestyle='--', label=f'90% → {n_90} componentes')
    plt.axhline(0.95, color='red',    linestyle='--', label=f'95% → {n_95} componentes')
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza explicada acumulada')
    plt.title('Varianza explicada por PCA', fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()

    print(f'Componentes para 90% varianza: {n_90}')
    print(f'Componentes para 95% varianza: {n_95}')
    return n_90, n_95


def plot_pca_2d(X_2d: np.ndarray, y_train: np.ndarray,
                label_encoder, pca_2d: PCA, save_path: str = None) -> None:
    """
    Scatter plot de la proyección PCA en 2 dimensiones.

    Args:
        X_2d:          array (n_samples, 2) con la proyección PCA
        y_train:       etiquetas numéricas del conjunto de train
        label_encoder: LabelEncoder ajustado
        pca_2d:        objeto PCA de 2 componentes (para % varianza en ejes)
        save_path:     si se indica, guarda la figura en esa ruta
    """
    plt.figure(figsize=(8, 6))
    for cls in np.unique(y_train):
        mask = y_train == cls
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            label=label_encoder.classes_[cls],
            alpha=0.7, color=TUMOR_COLORS[cls], s=30
        )
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Proyección PCA 2D — Train', fontsize=13)
    plt.legend(title='Tumor')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_pca_loadings(pca: PCA, gene_names: list, n: int = 20, save_path: str = None) -> None:
    """
    Barplot horizontal con los N genes de mayor contribución a PC1 y PC2.

    Args:
        pca:        objeto PCA ya ajustado
        gene_names: lista con los nombres de los genes (columnas de X)
        n:          número de genes a mostrar por componente (default 20)
        save_path:  si se indica, guarda la figura en esa ruta
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ax in enumerate(axes):
        loadings = pca.components_[i]
        top_idx  = np.argsort(np.abs(loadings))[-n:][::-1]
        top_genes = [gene_names[j] for j in top_idx]
        top_vals  = loadings[top_idx]
        colors_bar = ['#4C72B0' if v > 0 else '#C44E52' for v in top_vals]
        ax.barh(top_genes[::-1], top_vals[::-1], color=colors_bar[::-1], alpha=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f'Top {n} genes — PC{i+1}', fontsize=12)
        ax.set_xlabel('Loading')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# 4. VISUALIZACIÓN UMAP
# ──────────────────────────────────────────────

def plot_umap(X_umap: np.ndarray, y_train: np.ndarray,
              label_encoder, title: str = 'Proyección UMAP 2D',
              save_path: str = None) -> None:
    """
    Scatter plot de la proyección UMAP en 2 dimensiones (clasificación supervisada).

    Args:
        X_umap:        array (n_samples, 2) con la proyección UMAP
        y_train:       etiquetas numéricas
        label_encoder: LabelEncoder ajustado
        title:         título del gráfico
        save_path:     si se indica, guarda la figura en esa ruta
    """
    plt.figure(figsize=(9, 7))
    for cls in np.unique(y_train):
        mask = y_train == cls
        plt.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            label=label_encoder.classes_[cls],
            alpha=0.7, color=TUMOR_COLORS[cls], s=30
        )
    plt.title(title, fontsize=13)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Tumor')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_umap_clusters(X_umap: np.ndarray, clusters: np.ndarray,
                       title: str = 'Clusters UMAP', save_path: str = None) -> None:
    """
    Scatter plot UMAP coloreado por clusters KMeans (análisis no supervisado BRCA).

    Args:
        X_umap:    array (n_samples, 2) con la proyección UMAP
        clusters:  array con las etiquetas de cluster (enteros)
        title:     título del gráfico
        save_path: si se indica, guarda la figura en esa ruta
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=clusters, cmap='Set1', alpha=0.7, s=40
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(title, fontsize=13)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# 5. EVALUACIÓN DE MODELOS
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray,
                          class_names: list, save_path: str = None) -> None:
    """
    Matriz de confusión con ConfusionMatrixDisplay.

    Args:
        y_test:      etiquetas reales (numéricas)
        y_pred:      etiquetas predichas (numéricas)
        class_names: lista con los nombres de las clases (le.classes_)
        save_path:   si se indica, guarda la figura en esa ruta
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Matriz de Confusión — Logistic Regression (Test Set)', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results: dict, save_path: str = None) -> None:
    """
    Barplot horizontal comparando la accuracy de todos los modelos (CV).

    Args:
        results:   dict {nombre_modelo: array_scores_cv}
        save_path: si se indica, guarda la figura en esa ruta
    """
    names  = list(results.keys())
    means  = [results[n].mean() for n in names]
    stds   = [results[n].std()  for n in names]
    colors = ['#4C72B0', '#55A868', '#DD8452', '#8172B2', '#C44E52']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, means, xerr=stds,
                   color=colors[:len(names)], capsize=5,
                   alpha=0.85, edgecolor='white')
    ax.set_xlabel('Accuracy (CV 5-fold)')
    ax.set_title('Comparación de clasificadores\n(datos reducidos con PCA)', fontsize=13)
    ax.set_xlim(0.85, 1.02)
    for bar, mean in zip(bars, means):
        ax.text(mean + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{mean:.4f}', va='center', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# 6. GUARDADO DE MODELOS
# ──────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """
    Guarda un modelo entrenado en disco con joblib.

    Args:
        model: objeto modelo de scikit-learn ya entrenado
        path:  ruta donde guardar el archivo .pkl
    
    Ejemplo:
        save_model(best_clf, 'src/model/production/logistic_regression_pca.pkl')
    """
    import joblib
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f'Modelo guardado en: {path}')


def load_model(path: str):
    """
    Carga un modelo guardado con joblib.

    Args:
        path: ruta al archivo .pkl

    Returns:
        Modelo cargado
    """
    import joblib
    model = joblib.load(path)
    print(f'Modelo cargado desde: {path}')
    return model
