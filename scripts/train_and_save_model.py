from sklearn.model_selection import train_test_split # Dividir os dados do conjunto em proporção de treinamento e testes
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz # Usar o modelo de árvore de decisão para classificação e desenhar a árvore
from sklearn.metrics import classification_report, confusion_matrix # Avaliar o desempenho do modelo
import pandas as pd
import pickle
import os

def train_and_save(path="app_backend/model/penguim_classifier_tree_model.pkl"):

    # CARREGA O CONJUNTO DE DADOS DE PINGUINS
    df_penguins_size = pd.read_csv('db/penguins_size.csv', sep=',')
    df_penguins_size = df_penguins_size.dropna()

    # TRATAMENTO DE DADOS (RETIRADA DAS COLUNAS NÃO NUMÉRICAS)
    X = df_penguins_size.drop(['species', 'island', 'sex'], axis=1)
    y = df_penguins_size['species']

    # VARIAVEIS DE TREINO E TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=3
    )

    # # Modelo utilizado:Tree Decision Classifier
    # TREINAMENTO DO MODELO
    tree_decision_classifier = DecisionTreeClassifier(random_state=0, criterion='gini')
    tree_decision_classifier = tree_decision_classifier.fit(X_train, y_train)

    # Salva no formato pickle
    with open(path, "wb") as f:
        pickle.dump(tree_decision_classifier, f)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Modelo treinado e salvo em: {path}")

if __name__ == "__main__":
    train_and_save()
