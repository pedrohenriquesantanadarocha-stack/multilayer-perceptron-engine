import numpy as np

from training import Training
from models.multilayerPerceptron import MultilayerPerceptron
from problems.controllerProblems import ControllerProblems
from hiperparameter.hiperparameterController import HiperparameterController
from graph.graph import plot_confusion_matrix, make_graph

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
        
        # Salva Hiperparâmetros e Pesos Iniciais
        hiper_controller.save_hiperparameters()
        model.save_weights("pesos_iniciais.npz")
        
        # Treinamento com Validação e Early Stopping
        trainer = Training(model, learning_rate=learning_rate, epochs=max_epochs, patience=paciencia)
        erros_treino, erros_val = trainer.fit(x_train, y_train, x_val, y_val)
        
        # Salva Erros e Pesos Finais
        np.savetxt("results/erros_treinamento.csv", erros_treino, delimiter=",")
        model.save_weights("results/pesos_finais.npz")
        
        # Avaliação no Conjunto de Teste
        predictions = model.forward(x_test)
        np.savetxt("results/saidas_teste.csv", predictions, delimiter=",")
        
        erro_teste = model.compute_loss(predictions, y_test)
        print(f"\nErro Final no Conjunto de Teste (MSE): {erro_teste:.6f}")
        
        # --- GERAÇÃO DE GRÁFICOS (Matplotlib permitido) ---
        make_graph(erros_treino, erros_val)

        # Matriz de Confusão sem sklearn
        # Extrai os índices das classes caso seja One-Hot Encoding
        y_test_classes = np.argmax(y_test, axis=1) if y_test.shape[1] > 1 else y_test.flatten()
        pred_classes = np.argmax(predictions, axis=1) if predictions.shape[1] > 1 else np.round(predictions).flatten()
        
        # Plota a matriz
        plot_confusion_matrix(y_test_classes, pred_classes)
        
        if input("\nDeseja testar outro problema? (s/n): ").lower() != 's':
            break