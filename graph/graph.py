import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd # Se não tiver, instale com: pip install pandas

def plot_confusion_matrix(y_true, y_pred):
    """
    Gera e plota a matriz de confusão usando apenas Numpy e Matplotlib.
    """
    num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão - Conjunto de Teste")
    plt.colorbar()

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig("results/matriz_confusao.png")
    plt.show(block=False)
    print("Matriz de confusão salva como 'results/matriz_confusao.png'.")

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

def plot_comparativo_neuronios(filename):
    """
    Gera um gráfico de barras comparando o número de épocas necessárias para 
    o Early Stopping parar a rede, variando o tamanho da camada oculta.
    """
    hidden_sizes = []
    epocas_para_convergir = []
    
    # Abre o arquivo CSV para ler o conteúdo real
    with open(filename, 'r') as file:
        linhas = file.readlines()
        
        # Pula o cabeçalho (primeira linha)
        linhas_dados = linhas[1:]
        
        for linha in linhas_dados:
            valores = linha.strip().split(",")
            
            # Garante que a linha não está vazia
            if len(valores) >= 2: 
                hidden_sizes.append(str(valores[0]) + " Neurônios") 
                epocas_para_convergir.append(int(valores[1]))

    # Verifica se há dados antes de plotar
    if not epocas_para_convergir:
        print(f"Aviso: Nenhum dado encontrado em {filename}")
        return

    # Criação do Gráfico
    plt.figure(figsize=(10, 6))
    barras = plt.bar(hidden_sizes, epocas_para_convergir, color='teal')
    
    plt.title('Impacto do Tamanho da Camada Oculta no Tempo de Treinamento')
    plt.xlabel('Arquitetura (Tamanho da Camada Oculta)')
    plt.ylabel('Épocas até Convergir (Early Stopping)')
    
    # Adiciona os números exatos em cima de cada barra
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, yval + (max(epocas_para_convergir)*0.01), 
                 int(yval), ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/grafico_comparativo_neuronios.png")
    plt.show(block=False)
    
def save_results_to_csv(filename, hidden_size, epocas_para_convergir, learning_rate, paciencia):
    """
    Salva os resultados de um treinamento no arquivo CSV (modo append para acumular dados).
    """
    # Cria o arquivo com cabeçalho se não existir
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("Hidden Size,Epochs to Converge,Learning Rate,Patience\n")
    
    # Adiciona os dados (append mode)
    with open(filename, "a") as f:
        f.write(f"{hidden_size},{epocas_para_convergir},{learning_rate},{paciencia}\n")

def delete_results_file(filename):
    """
    Deleta o arquivo de resultados se ele existir, para começar do zero.
    """
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Arquivo {filename} deletado para iniciar novos testes.")

def gerar_tabela_desempenho(filename):
    # Carrega os dados
    df = pd.read_csv(filename, names=['Neurônios', 'LR', 'Épocas', 'Tempo (s)'])
    
    # Formata a tabela para ficar bonita
    fig, ax = plt.subplots(figsize=(8, 3)) # Tamanho da figura da tabela
    ax.axis('off') # Esconde os eixos do gráfico
    
    # Cria a tabela
    tabela = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    
    # Ajusta o estilo
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(12)
    tabela.scale(1.2, 1.5) # Ajusta o espaçamento das células
    
    plt.title("Resumo do Desempenho dos Experimentos")
    plt.savefig("results/tabela_desempenho.png", bbox_inches='tight')
    plt.show(block=False)

def save_performance_overview_to_csv(filename, tempo_total_treino, learning_rate, hidden_size, epocas_para_convergir):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("Neuronios,LR,Epocas,Tempo\n")
    
    with open(filename, "a") as f:
        f.write(f"{hidden_size},{learning_rate},{len(epocas_para_convergir)},{tempo_total_treino:.4f}\n")
