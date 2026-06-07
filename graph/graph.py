import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_true, y_pred):
    """
    Gera e plota a matriz de confusão usando apenas Numpy e Matplotlib.
    """
    # Adicionamos o int() aqui para garantir que o número de classes seja inteiro (resolve o erro!)
    num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Preenche a matriz de confusão (adicionamos int() em t e p para evitar que índices sejam floats)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão - Conjunto de Teste")
    plt.colorbar()

    # Adiciona os números dentro dos quadrados
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig("matriz_confusao.png")
    
    # Substituímos o plt.show() por plt.close() para não travar o terminal!
    plt.show(block=False)
    print("Matriz de confusão salva como 'matriz_confusao.png'.")

def make_graph(erros_treino, erros_val):
    """
    Gera um gráfico de convergência do erro ao longo das épocas usando Matplotlib.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(erros_treino, label='Erro de Treinamento')
    if erros_val:
        plt.plot(erros_val, label='Erro de Validação')
    plt.title("Convergência do Erro (MSE) por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Quadrático Médio (MSE)")
    plt.legend()
    plt.grid()
    plt.savefig("results/grafico_erros.png")
    plt.show(block=False)