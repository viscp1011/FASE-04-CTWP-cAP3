FIAP - Faculdade de Informática e Administração Paulista
FIAP - Faculdade de Informática e Admnistração Paulista


Nome do grupo
👨‍🎓 Integrante:
Vitório Paciulo

👩‍🏫 Professores:

Tutor(a):
Ana Cristina dos Santos

Coordenador(a):
André Godoi Chiovato

# 🌾 Classificação de Grãos de Trigo com Aprendizado de Máquina (Seeds Dataset – CRISP-DM)

Este projeto aplica **aprendizado de máquina** para classificar **variedades de trigo** a partir de suas **características físicas**, utilizando o famoso **Seeds Dataset** (UCI Machine Learning Repository) e a metodologia **CRISP-DM**.

O contexto é o de **cooperativas agrícolas de pequeno porte**, onde a classificação de grãos costuma ser feita de forma **manual**, por especialistas, o que pode ser:

- Demorado  
- Sujeito a erro humano  
- Pouco padronizado entre avaliadores  

A ideia é mostrar que, com um conjunto de dados relativamente simples e algoritmos clássicos de ML, é possível **automatizar (ou apoiar) a classificação** com alta acurácia.

---

## 🎯 Objetivos do Projeto

1. Aplicar a metodologia **CRISP-DM** no problema de classificação de grãos.
2. **Analisar e pré-processar** o Seeds Dataset.
3. **Implementar e comparar** diferentes algoritmos de classificação:
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Random Forest  
   - Regressão Logística  
   - Naive Bayes  
4. **Otimizar** os modelos com **GridSearchCV**.
5. **Interpretar os resultados** e extrair insights relevantes, relacionando-os com o contexto real de cooperativas agrícolas.
6. Discutir **por que os resultados podem ser considerados confiáveis**.

---

## 📁 Dataset: Seeds (UCI Machine Learning Repository)

- Fonte: UCI Machine Learning Repository – *Seeds Dataset*  
- Número de amostras: **210**  
- Classes (3 variedades de trigo, 70 amostras cada):
  - `1` → **Kama**  
  - `2` → **Rosa**  
  - `3` → **Canadian**  

### 🔢 Atributos (features)

Cada amostra é um grão de trigo descrito por 7 características físicas, extraídas a partir do contorno do grão:

1. **area** – Área do grão.
2. **perimeter** – Comprimento do contorno do grão.
3. **compactness** – Compacidade:  
   \\( C = \frac{4 \pi A}{P^2} \\)  
   Mede o quão “cheio”/compacto é o grão. Valores menores indicam formatos mais alongados.
4. **kernel_length** – Comprimento do núcleo (eixo maior da elipse equivalente).
5. **kernel_width** – Largura do núcleo (eixo menor da elipse).
6. **asymmetry** – Coeficiente de assimetria; mensura o quanto o grão foge de um formato simétrico.
7. **groove_length** – Comprimento do sulco do núcleo (a “marquinha” central do grão).
8. **class** / **class_name** – Variedade de trigo (Kama, Rosa, Canadian).

O dataset é **balanceado**: 70 amostras de cada classe.

---

## 🧠 Metodologia: CRISP-DM

O projeto foi estruturado seguindo as fases do **CRISP-DM**:

### 1. Entendimento do Negócio

- Problema: classificação manual de grãos de trigo → lenta, subjetiva, mais cara.
- Objetivo: desenvolver um modelo de ML capaz de **classificar automaticamente** a variedade do grão com boa acurácia, servindo como:
  - Apoio ao especialista;
  - Ferramenta para acelerar o fluxo de trabalho na cooperativa.

### 2. Entendimento dos Dados

Passos principais realizados:

- Leitura do dataset `seeds_dataset.txt` (sem cabeçalho, separado por espaços).
- Nomeação das colunas com base na documentação da UCI.
- Análises iniciais:
  - `df.head()`, `df.info()`, `df.describe()`;
  - Verificação de **valores ausentes** (não há NaNs);
  - Distribuição das classes (`class` e `class_name`) → dataset balanceado.
- Análise exploratória:
  - Histogramas das 7 features;
  - Boxplots (para visualizar possíveis outliers);
  - Matriz de correlação;
  - Gráficos de dispersão:
    - `area × perimeter`;
    - `kernel_length × kernel_width`;  
    coloridos pela classe.

**Principais observações:**

- Forte correlação entre variáveis de “tamanho” (área, perímetro, comprimento e sulco).
- Compactness e assimetria ajudam a capturar o **formato**, não apenas o tamanho.
- As classes formam agrupamentos parcialmente separados no espaço das features → problema propício à classificação.

### 3. Preparação dos Dados

- Remoção apenas das colunas de rótulo para formar `X`:
  - `X` = todas as features numéricas (`area`, `perimeter`, …, `groove_length`).
  - `y` = `class` (1, 2 ou 3).
- Divisão em **treino** e **teste**:
  - 70% treino, 30% teste;
  - `train_test_split(..., stratify=y, random_state=42)`  
    → garante mesma proporção de classes em treino e teste.
- Padronização:
  - Uso de `StandardScaler` dentro de um **Pipeline** (`scikit-learn`);
  - Evita vazamento de informação (data leakage) e melhora desempenho de SVM e KNN.

### 4. Modelagem

Foram treinados 5 modelos:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Regressão Logística**
- **Naive Bayes**

Cada modelo foi treinado em um `Pipeline`:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", <modelo>)
])
