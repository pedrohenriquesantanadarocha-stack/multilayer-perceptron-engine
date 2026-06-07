import numpy as np

class Training:
    def __init__(self, model, learning_rate=0.09, epochs=100000000000):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, x, y):
        historico_erros = []
        taxa_inicial = self.learning_rate
        fator_decaimento = 0.999
        
        print("Iniciando treinamento Estocástico (Linha por Linha)...")
        for epoch in range(self.epochs):
            ##lr_dinamica = taxa_inicial * (fator_decaimento ** epoch)
            
            erro_acumulado_epoca = 0
            
            # MODO ESTOCÁSTICO: Atualiza os pesos linha por linha!
            for i in range(len(x)):
                X_linha = x[i:i+1] # Pega apenas 1 linha de X
                y_linha = y[i:i+1] # Pega apenas 1 linha de Y
                
                # Forward, Loss e Backward para UMA única linha
                output = self.model.forward(X_linha)
                erro_instantaneo = self.model.compute_loss(output, y_linha)
                erro_acumulado_epoca += erro_instantaneo
                
                self.model.backward(X_linha, y_linha, taxa_inicial)
            
            # O erro da época é a média dos erros das linhas
            erro_medio_epoca = erro_acumulado_epoca / len(x)
            historico_erros.append(erro_medio_epoca)
            
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                print(f"Época {epoch}/{self.epochs} - Erro: {erro_medio_epoca:.6f} | LR: {taxa_inicial:.6f}")
                
        return historico_erros