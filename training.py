import numpy as np

class Training:
    def __init__(self, model, learning_rate=0.1, epochs=1000, patience=50):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience # Paciência para a Parada Antecipada

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        historico_erros_treino = []
        historico_erros_val = []
        
        melhor_erro_val = float('inf')
        epocas_sem_melhoria = 0
        
        print("Iniciando treinamento Estocástico com Parada Antecipada...")
        for epoch in range(self.epochs):
            erro_acumulado_epoca = 0
            
            # MODO ESTOCÁSTICO: Atualiza os pesos linha por linha
            for i in range(len(x_train)):
                X_linha = x_train[i:i+1]
                y_linha = y_train[i:i+1]
                
                output = self.model.forward(X_linha)
                erro_instantaneo = self.model.compute_loss(output, y_linha)
                erro_acumulado_epoca += erro_instantaneo
                
                self.model.backward(X_linha, y_linha, self.learning_rate)
            
            # Erro Médio de Treinamento
            erro_medio_epoca = erro_acumulado_epoca / len(x_train)
            historico_erros_treino.append(erro_medio_epoca)
            
            # Cálculo do Erro de Validação e Parada Antecipada
            if x_val is not None and y_val is not None:
                val_output = self.model.forward(x_val)
                erro_val = self.model.compute_loss(val_output, y_val)
                historico_erros_val.append(erro_val)
                
                if erro_val < melhor_erro_val:
                    melhor_erro_val = erro_val
                    epocas_sem_melhoria = 0
                    # Você poderia salvar os melhores pesos aqui
                else:
                    epocas_sem_melhoria += 1
                
                if epocas_sem_melhoria >= self.patience:
                    print(f"Parada Antecipada ativada na época {epoch}! Erro de validação parou de cair.")
                    break
            
            if epoch % 50 == 0 or epoch == self.epochs - 1:
                if x_val is not None:
                    print(f"Época {epoch}/{self.epochs} - Erro Treino: {erro_medio_epoca:.6f} | Erro Val: {erro_val:.6f}")
                else:
                    print(f"Época {epoch}/{self.epochs} - Erro Treino: {erro_medio_epoca:.6f}")
                
        return historico_erros_treino, historico_erros_val
    
    @staticmethod
    def train_val_test_split(x, y, val_ratio=0.15, test_ratio=0.15):
        """
        Divide os dados manualmente usando apenas Numpy. 
        Blinda o código contra datasets muito pequenos (ex: portas lógicas).
        """
        # Se o dataset for minúsculo (como XOR, AND, OR que têm 4 linhas),
        # não faz sentido dividir. Retornamos o próprio dataset para tudo.
        if len(x) < 10:
            print("\n[Aviso] Dataset muito pequeno detectado! Usando os mesmos dados para Treino, Validação e Teste para não quebrar a execução.")
            return x, y, x, y, x, y

        np.random.seed(42) # Semente fixa para reprodutibilidade
        indices = np.random.permutation(len(x))
        
        # O max(1, ...) garante que nunca teremos um conjunto de tamanho 0
        test_size = max(1, int(len(x) * test_ratio))
        val_size = max(1, int(len(x) * val_ratio))
        
        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size+val_size]
        train_idx = indices[test_size+val_size:]
        
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx]