#%% ============================
# 1. IMPORTAÇÕES E CONFIGURAÇÕES
#===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Estilo de gráficos (sem usar plt.style.use("seaborn"))
sns.set_theme(style="darkgrid", font_scale=1.1)


#%% ============================
# 2. CARREGAR O DATASET
#===============================

# CAMINHO COMPLETO DO DATASET (Desktop)
file_path = r"C:\Users\Vitorio\Desktop\seeds_dataset.txt"

# Dataset da UCI não tem cabeçalho e é separado por espaços
df = pd.read_csv(file_path, delim_whitespace=True, header=None)

# Nome das colunas conforme a descrição da UCI
df.columns = [
    "area",            # 1. area A
    "perimeter",       # 2. perimeter P
    "compactness",     # 3. compactness C = 4*pi*A / P^2
    "kernel_length",   # 4. length of kernel
    "kernel_width",    # 5. width of kernel
    "asymmetry",       # 6. asymmetry coefficient
    "groove_length",   # 7. length of kernel groove
    "class"            # 8. tipo do trigo (1,2,3)
]

# Mapa das classes para nomes
class_map = {1: "Kama", 2: "Rosa", 3: "Canadian"}
df["class_name"] = df["class"].map(class_map)

print("Primeiras linhas do dataset:")
print(df.head())


#%% ============================
# 3. ESTATÍSTICAS DESCRITIVAS E DADOS FALTANTES
#===============================

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe().T)

print("\nDistribuição das classes (numéricas):")
print(df["class"].value_counts())

print("\nDistribuição das classes (nomes):")
print(df["class_name"].value_counts())

print("\nValores ausentes por coluna:")
print(df.isna().sum())

# Se houver NaN, preenche com a mediana de cada coluna numérica
df[df.columns] = df[df.columns].fillna(df.median(numeric_only=True))

print("\nValores ausentes após tratamento:")
print(df.isna().sum())


#%% ============================
# 4. VISUALIZAÇÃO EXPLORATÓRIA
#===============================

df_features = df.drop(columns=["class", "class_name"])

# Histogramas
df_features.hist(bins=15, figsize=(12, 8))
plt.suptitle("Distribuição das Características (Histograma)", fontsize=16)
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_features)
plt.title("Boxplots das Características")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Matriz de correlação
plt.figure(figsize=(10, 8))
corr = df_features.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação entre as Features")
plt.tight_layout()
plt.show()

# Scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(
    x="area", y="perimeter",
    hue="class_name", data=df
)
plt.title("Área vs Perímetro")

plt.subplot(1, 2, 2)
sns.scatterplot(
    x="kernel_length", y="kernel_width",
    hue="class_name", data=df
)
plt.title("Comprimento vs Largura do Núcleo")

plt.tight_layout()
plt.show()


#%% ============================
# 5. SEPARAÇÃO TREINO/TESTE
#===============================

X = df_features.values
y = df["class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Formato X_train:", X_train.shape)
print("Formato X_test :", X_test.shape)


#%% ============================
# 6. FUNÇÃO PARA TREINAR E AVALIAR MODELOS
#===============================

def treinar_avaliar_modelo(nome, modelo, X_train, X_test, y_train, y_test):
    """
    Cria um pipeline com StandardScaler + modelo,
    treina e imprime métricas e matriz de confusão.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", modelo)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\n" + "="*60)
    print(f"Modelo: {nome}")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Kama", "Rosa", "Canadian"]
    ))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Kama", "Rosa", "Canadian"],
        yticklabels=["Kama", "Rosa", "Canadian"]
    )
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.tight_layout()
    plt.show()

    return pipe, y_pred


#%% ============================
# 7. MODELOS BASELINE (KNN, SVM, RF, LR, NB)
#===============================

modelos_baseline = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, multi_class="ovr"
    ),
    "NaiveBayes": GaussianNB()
}

pipes_baseline = {}
for nome, modelo in modelos_baseline.items():
    pipe, _ = treinar_avaliar_modelo(
        nome, modelo, X_train, X_test, y_train, y_test
    )
    pipes_baseline[nome] = pipe


#%% ============================
# 8. OTIMIZAÇÃO DE HIPERPARÂMETROS – KNN
#===============================

param_grid_knn = {
    "clf__n_neighbors": [3, 5, 7, 9],
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["euclidean", "manhattan"]
}

pipe_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])

grid_knn = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid_knn,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_knn.fit(X_train, y_train)

print("\nMelhores hiperparâmetros KNN:", grid_knn.best_params_)
print("Melhor acurácia média (cv) KNN:", grid_knn.best_score_)

y_pred_knn = grid_knn.predict(X_test)
print("\nDesempenho KNN Otimizado no Teste:")
print("Acurácia:", accuracy_score(y_test, y_pred_knn))
print(classification_report(
    y_test, y_pred_knn,
    target_names=["Kama", "Rosa", "Canadian"]
))

cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(4, 4))
sns.heatmap(
    cm_knn, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Kama", "Rosa", "Canadian"],
    yticklabels=["Kama", "Rosa", "Canadian"]
)
plt.title("Matriz de Confusão - KNN (Otimizado)")
plt.tight_layout()
plt.show()


#%% ============================
# 9. OTIMIZAÇÃO – SVM
#===============================

param_grid_svm = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto"],
    "clf__kernel": ["rbf", "linear"]
}

pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC())
])

grid_svm = GridSearchCV(
    estimator=pipe_svm,
    param_grid=param_grid_svm,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_svm.fit(X_train, y_train)

print("\nMelhores hiperparâmetros SVM:", grid_svm.best_params_)
print("Melhor acurácia média (cv) SVM:", grid_svm.best_score_)

y_pred_svm = grid_svm.predict(X_test)
print("\nDesempenho SVM Otimizado no Teste:")
print("Acurácia:", accuracy_score(y_test, y_pred_svm))
print(classification_report(
    y_test, y_pred_svm,
    target_names=["Kama", "Rosa", "Canadian"]
))

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(4, 4))
sns.heatmap(
    cm_svm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Kama", "Rosa", "Canadian"],
    yticklabels=["Kama", "Rosa", "Canadian"]
)
plt.title("Matriz de Confusão - SVM (Otimizado)")
plt.tight_layout()
plt.show()


#%% ============================
# 10. OTIMIZAÇÃO – RANDOM FOREST
#===============================

param_grid_rf = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 4],
    "clf__min_samples_leaf": [1, 2]
}

pipe_rf = Pipeline([
    ("scaler", StandardScaler()),  # RF não precisa, mas não atrapalha
    ("clf", RandomForestClassifier(random_state=42))
])

grid_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

print("\nMelhores hiperparâmetros RF:", grid_rf.best_params_)
print("Melhor acurácia média (cv) RF:", grid_rf.best_score_)

y_pred_rf = grid_rf.predict(X_test)
print("\nDesempenho Random Forest Otimizado no Teste:")
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print(classification_report(
    y_test, y_pred_rf,
    target_names=["Kama", "Rosa", "Canadian"]
))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(4, 4))
sns.heatmap(
    cm_rf, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Kama", "Rosa", "Canadian"],
    yticklabels=["Kama", "Rosa", "Canadian"]
)
plt.title("Matriz de Confusão - Random Forest (Otimizado)")
plt.tight_layout()
plt.show()
