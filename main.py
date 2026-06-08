import numpy as np
import time

from training import Training
from models.multilayerPerceptron import MultilayerPerceptron
from problems.controllerProblems import ControllerProblems
from hiperparameter.hiperparameterController import HiperparameterController
from graph.graph import gerar_tabela_desempenho, plot_confusion_matrix, make_graph, plot_comparativo_neuronios, save_performance_overview_to_csv, save_results_to_csv, delete_results_file

np.random.seed(42)

if __name__ == "__main__":
    controller = ControllerProblems()

    while True:
        x, y = controller.select_problem_interactively()

        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # Parâmetros definidos interativamente via entrada de dados
        hiper_controller = HiperparameterController()
        hidden_size, learning_rate, max_epochs, paciencia = hiper_controller.get_hyperparameters()
        hiper_controller.set_hiperparameters(input_size=x.shape[1], hidden_size=hidden_size, output_size=y.shape[1], learning_rate=learning_rate, max_epochs=max_epochs)
        
        # Divisão de Dados puramente com Numpy
        x_train, y_train, x_val, y_val, x_test, y_test = Training.train_val_test_split(x, y)
        
        input_size = x.shape[1]
        output_size = y.shape[1]
        
        print(f"\nCriando rede com {input_size} entradas, {hidden_size} na camada oculta e {output_size} saídas...")
        model = MultilayerPerceptron(input_size, hidden_size, output_size)
        
        #Cria os pesos inicialmente aleatórios
        W1, b1, W2, b2 = model.make_weights()

        # Salva Hiperparâmetros e Pesos Iniciais
        hiper_controller.save_hiperparameters()
        model.save_weights(W1, b1, W2, b2)
        
        # Treinamento com Validação e Early Stopping
        trainer = Training(model, learning_rate=learning_rate, epochs=max_epochs, patience=paciencia)

        inicio_treino = time.time()
        erros_treino, erros_val = trainer.fit(x_train, y_train, x_val, y_val)
        fim_treino = time.time()

        tempo_total_treino = fim_treino - inicio_treino
        print(f"\nTempo total de treinamento: {tempo_total_treino:.2f} segundos.")

        save_performance_overview_to_csv("results/desempenho_geral.csv", tempo_total_treino, learning_rate, hidden_size, epocas_para_convergir=erros_treino)
        gerar_tabela_desempenho("results/desempenho_geral.csv")

        epocas_para_convergir = len(erros_treino)
        print(f"\nTreinamento finalizado em {epocas_para_convergir} épocas.")
        
        # Salva Erros e Pesos Finais
        np.savetxt("results/erros_treinamento.csv", erros_treino, delimiter=",")
        model.save_weights_files("results/pesos_finais.npz")
        
        # Avaliação no Conjunto de Teste
        predictions = model.feed_forward(x_test)
        np.savetxt("results/saidas_teste.csv", predictions, delimiter=",")
        
        erro_teste = model.compute_loss(predictions, y_test)
        print(f"\nErro Final no Conjunto de Teste (MSE): {erro_teste:.6f}")
        
        # --- GERAÇÃO DE GRÁFICOS (Matplotlib permitido) ---
        make_graph(erros_treino, erros_val)

        # Extrai os índices das classes caso seja One-Hot Encoding

        y_test_classes = trainer.extract_classes(y_test)
        pred_classes = trainer.extract_classes(predictions)

        # Plota a matriz
        plot_confusion_matrix(y_test_classes, pred_classes)
        save_results_to_csv("results/matriz_confusao.csv", hidden_size, epocas_para_convergir, learning_rate, paciencia)
        plot_comparativo_neuronios("results/matriz_confusao.csv") # Substitua pelos resultados reais obtidos

        if input("\nDeseja testar outro problema? (s/n): ").lower() != 's':
            delete_results_file("results/matriz_confusao.csv")
            delete_results_file("results/desempenho_geral.csv")
            break