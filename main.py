import numpy as np
import time

from training import Training
from models.multilayerPerceptron import MultilayerPerceptron
from problems.controllerProblems import ControllerProblems
from hiperparameter.hiperparameterController import HiperparameterController
from graph.graph import gerar_tabela_desempenho, plot_confusion_matrix, make_graph, plot_comparativo_neuronios, save_performance_overview_to_csv, save_results_to_csv, delete_results_file

# Define a semente para garantir a reprodutibilidade dos resultados aleatórios
np.random.seed(42)

if __name__ == "__main__":
    # Inicializa o controlador para seleção de datasets/problemas
    controllerProblems = ControllerProblems()

    while True:
        # Seleciona e carrega os dados do problema escolhido
        x, y = controllerProblems.select_problem_interactively()

        # Ajusta a dimensionalidade dos dados caso não estejam em formato tabular (2D)
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # Coleta parâmetros de configuração do usuário via interface
        hiper_controller = HiperparameterController()
        hidden_size, learning_rate, max_epochs, paciencia = hiper_controller.get_hyperparameters()
        hiper_controller.set_hiperparameters(input_size=x.shape[1], hidden_size=hidden_size, output_size=y.shape[1], learning_rate=learning_rate, max_epochs=max_epochs)
        
        # Realiza a divisão dos dados em conjuntos de treino, validação e teste
        x_train, y_train, x_val, y_val, x_test, y_test = Training.train_val_test_split(x, y)
        
        input_size = x.shape[1]
        output_size = y.shape[1]
        
        # Instancia o modelo Multilayer Perceptron
        print(f"\nCriando rede com {input_size} entradas, {hidden_size} na camada oculta e {output_size} saídas...")
        model = MultilayerPerceptron(input_size, hidden_size, output_size)
        
        # Inicializa os pesos e vieses da rede de forma aleatória
        W1, b1, W2, b2 = model.make_weights()

        # Armazena os hiperparâmetros e os pesos iniciais em arquivo
        hiper_controller.save_hiperparameters()
        model.save_weights(W1, b1, W2, b2)
        
        # Configura o objeto de treinamento com os parâmetros definidos
        trainer = Training(model, learning_rate=learning_rate, epochs=max_epochs, patience=paciencia)

        # Executa o processo de treinamento e mede o tempo de processamento
        inicio_treino = time.time()
        erros_treino, erros_val = trainer.fit(x_train, y_train, x_val, y_val)
        fim_treino = time.time()

        tempo_total_treino = fim_treino - inicio_treino
        print(f"\nTempo total de treinamento: {tempo_total_treino:.2f} segundos.")

        # Registra métricas de desempenho e atualiza a tabela geral de resultados
        save_performance_overview_to_csv("results/desempenho_geral.csv", tempo_total_treino, learning_rate, hidden_size, epocas_para_convergir=erros_treino)
        gerar_tabela_desempenho("results/desempenho_geral.csv")

        epocas_para_convergir = len(erros_treino)
        print(f"\nTreinamento finalizado em {epocas_para_convergir} épocas.")
        
        # Persiste o histórico de erros e os pesos finais após convergência
        np.savetxt("results/erros_treinamento.csv", erros_treino, delimiter=",")
        model.save_weights_files("results/pesos_finais.npz")
        
        # Realiza a predição no conjunto de teste e calcula o erro (MSE)
        predictions = model.feed_forward(x_test)
        np.savetxt("results/saidas_teste.csv", predictions, delimiter=",")
        
        erro_teste = model.compute_loss(predictions, y_test)
        print(f"\nErro Final no Conjunto de Teste (MSE): {erro_teste:.6f}")
        
        # Gera visualizações gráficas da convergência do treinamento
        make_graph(erros_treino, erros_val)

        # Converte as saídas de probabilidade/One-Hot para índices de classes para análise de erros
        y_test_classes = trainer.extract_classes(y_test)
        pred_classes = trainer.extract_classes(predictions)

        # Gera a matriz de confusão e o comparativo de desempenho
        plot_confusion_matrix(y_test_classes, pred_classes)
        save_results_to_csv("results/matriz_confusao.csv", hidden_size, epocas_para_convergir, learning_rate, paciencia)
        plot_comparativo_neuronios("results/matriz_confusao.csv")

        # Verifica se o usuário deseja realizar um novo teste ou encerrar o programa
        if input("\nDeseja testar outro problema? (s/n): ").lower() != 's':
            # Limpa arquivos temporários de resultados antes de fechar
            delete_results_file("results/matriz_confusao.csv")
            delete_results_file("results/desempenho_geral.csv")
            break