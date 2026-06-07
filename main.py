from training import Training
from models.multilayerPerceptron import MultilayerPerceptron
from Problems.problems import Problems
from Problems.controllerProblems import ControllerProblems

if __name__ == "__main__":
    controller = ControllerProblems()

    while True:
        # 1. PRIMEIRO carregamos os dados do problema
        x, y = controller.select_problem_interactively()
        
        # 2. DEPOIS definimos a arquitetura dinamicamente com base nos dados
        input_size = x.shape[1]  # O número de colunas de x é o input_size
        output_size = y.shape[1] # O número de colunas de y é o output_size
        hidden_size = 4 # Você pode ajustar esse hiperparâmetro depois
        
        print(f"\nCriando rede com {input_size} entradas e {output_size} saídas...")
        model = MultilayerPerceptron(input_size, hidden_size, output_size)
        
        # 3. Treinamento
        trainer = Training(model, learning_rate=0.1, epochs=100000)
        
        # O método fit agora vaiac retornar o histórico de erros
        erros_historico = trainer.fit(x, y)
        
        # 4. Avaliação (Predição nos mesmos dados por enquanto)
        predictions = model.forward(x)
        print("\nPredições finais (algumas amostras):")
        print(predictions[:5]) # Mostra apenas os 5 primeiros para não poluir a tela
        
        if input("\nDo you want to select another problem? (yes/no): ").lower() != 'yes':
            break