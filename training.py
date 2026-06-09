import numpy as np

class Training:
    """
    Classe responsável pelo controle do ciclo de vida de treinamento da rede neural.
    Gerencia a iteração sobre os dados, o cálculo dos gradientes através do modelo,
    a monitoração do erro (loss) e a estratégia de parada antecipada (Early Stopping).
    """
    def __init__(self, model, learning_rate=0.1, epochs=1000, patience=50):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        # A paciência define o número máximo de épocas em que o erro de validação pode 
        # estagnar antes de interrompermos o treino, prevenindo o sobreajuste (overfitting) 
        # ao parar o modelo quando ele perde a capacidade de generalização.
        self.patience = patience 

    def fit(self, x_train, y_train, x_val=None, y_val=None, min_delta=1e-4):
        historico_erros_treino = []
        historico_erros_val = []
        
        melhor_erro_val = float('inf')
        epocas_sem_melhoria = 0
        
        print("Iniciando treinamento Estocástico com Parada Antecipada...")
        for epoch in range(self.epochs):
            erro_acumulado_epoca = 0
            
            # MODO ESTOCÁSTICO: Atualiza os pesos linha por linha (uma amostra por vez).
            # Diferente do gradiente em lote (batch), este método introduz ruído no gradiente,
            # o que ajuda a escapar de mínimos locais e acelera a convergência em datasets grandes,
            # tornando o aprendizado mais dinâmico a cada exemplo apresentado à rede.
            for i in range(len(x_train)):
                X_linha = x_train[i:i+1]
                y_linha = y_train[i:i+1]
                
                output = self.model.feed_forward(X_linha)
                erro_instantaneo = self.model.compute_loss(output, y_linha)
                erro_acumulado_epoca += erro_instantaneo
                
                self.model.backward(X_linha, y_linha, self.learning_rate)
            
            # Calculamos a média do erro nesta época para avaliar a tendência do aprendizado.
            # Um erro decrescente indica que os pesos estão sendo ajustados corretamente na
            # direção que minimiza a função de custo global do modelo.
            erro_medio_epoca = erro_acumulado_epoca / len(x_train)
            historico_erros_treino.append(erro_medio_epoca)

            # Cálculo do Erro de Validação e Parada Antecipada (Early Stopping).
            # Avaliamos o desempenho em um conjunto de dados que a rede não viu durante o ajuste.
            # Se o erro de validação para de cair, significa que a rede está começando a decorar
            # os dados de treino em vez de aprender o padrão, sendo o momento ideal para parar.
            if x_val is not None and y_val is not None:
                val_output = self.model.feed_forward(x_val)
                erro_val = self.model.compute_loss(val_output, y_val)
                historico_erros_val.append(erro_val)
                
                # APLICAÇÃO DO MIN_DELTA: Exige uma melhoria mínima real para resetar a paciência.
                if erro_val < (melhor_erro_val - min_delta):
                    melhor_erro_val = erro_val
                    epocas_sem_melhoria = 0
                else:
                    epocas_sem_melhoria += 1
                
                if epocas_sem_melhoria >= self.patience:
                    print(f"Parada Antecipada ativada na época {epoch}! Erro de validação estagnou.")
                    break
            
            # Log de acompanhamento do treinamento para monitorar a convergência do modelo.
            if epoch % 50 == 0 or epoch == self.epochs - 1:
                if x_val is not None:
                    print(f"Época {epoch}/{self.epochs} - Erro Treino: {erro_medio_epoca:.6f} | Erro Val: {erro_val:.6f}")
                else:
                    print(f"Época {epoch}/{self.epochs} - Erro Treino: {erro_medio_epoca:.6f}")
                
        return historico_erros_treino, historico_erros_val
    
    @staticmethod
    def train_val_test_split(x, y, val_ratio=0.15, test_ratio=0.15):
        """
        Divide os dados manualmente em conjuntos de treino, validação e teste.
        A separação é crucial para garantir uma avaliação imparcial: o treino ajusta os pesos,
        a validação guia os hiperparâmetros e a parada, e o teste final mede a capacidade 
        real de generalização do modelo em dados desconhecidos.
        """
        # Se o dataset for minúsculo (como portas lógicas XOR, AND, OR), a divisão falharia 
        # por falta de amostras suficientes. Nestes casos, retornamos o dataset completo, 
        # pois não há dados suficientes para uma separação estatística significativa.
        if len(x) < 10:
            print("\n[Aviso] Dataset muito pequeno detectado! Usando os mesmos dados para Treino, Validação e Teste.")
            return x, y, x, y, x, y

        # Utilizamos uma permutação aleatória dos índices para garantir que a distribuição 
        # dos dados nos três conjuntos seja homogênea e representativa, evitando que um 
        # conjunto contenha apenas exemplos de uma classe específica.
        np.random.seed(42)
        indices = np.random.permutation(len(x))
        
        # O uso de max(1, ...) assegura que, mesmo em datasets pequenos, cada conjunto 
        # possua pelo menos uma amostra, garantindo que o ciclo de treino não tente 
        # processar um conjunto vazio, o que causaria erros de indexação matemática.
        test_size = max(1, int(len(x) * test_ratio))
        val_size = max(1, int(len(x) * val_ratio))
        
        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size+val_size]
        train_idx = indices[test_size+val_size:]
        
        return x[train_idx], y[train_idx], x[val_idx], y[val_idx], x[test_idx], y[test_idx]
    
    def extract_classes(self, y):
        """
        Converte as saídas probabilísticas ou One-Hot em rótulos de classe inteiros.
        Para problemas de classificação, isso é fundamental para calcular métricas como 
        a matriz de confusão, onde precisamos comparar a classe real com a predição da rede.
        """
        # Se tivermos múltiplas colunas de saída (One-Hot), selecionamos o índice com o maior valor.
        # Se for uma saída binária única, arredondamos para 0 ou 1, transformando a saída 
        # contínua da sigmoide em uma decisão de classificação discreta.
        if y.shape[1] > 1: 
            return np.argmax(y, axis=1)
        else: 
            return np.round(y).flatten()