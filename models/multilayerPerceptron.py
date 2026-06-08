import numpy as np

class MultilayerPerceptron:
    """
    Implementação de uma Rede Neural Artificial Multilayer Perceptron (MLP) com uma camada oculta.
    Treinamento realizado através do algoritmo Backpropagation (Gradiente Descendente em Lote/Batch).
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Definição da arquitetura da rede (Hiperparâmetros)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicialização de Pesos (W) e Viéses (b - bias)
        # Os pesos são inicializados com valores aleatórios pequenos para quebrar a simetria 
        # e evitar que os neurônios aprendam as mesmas características.
        # As matrizes seguem a dimensão: (camada_anterior, camada_atual)
        
        # O bias é inicializado com zeros. Dimensão: (1, camada_atual)
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        
        self.b1 = np.zeros((1, hidden_size))
        
        # Pesos e bias da camada de saídas
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        """
        Função de ativação Sigmoide. 
        Mapeia qualquer valor real para um intervalo entre 0 e 1, introduzindo não-linearidade na rede.
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Derivada da função de ativação Sigmoide.
        Necessária para o cálculo do gradiente durante o algoritmo de Backpropagation.
        A derivada da sigmoide(x) é matematicamente igual a sigmoide(x) * (1 - sigmoide(x)).
        """
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def feed_forward(self, X):
        """
        Processo de Forward Pass (Propagação para frente).
        Calcula a saída da rede neural passando os dados de entrada camada por camada.
        """
        # Camada Oculta (Hidden Layer)
        # Z1 = Campo induzido (soma ponderada das entradas + bias)
        self.Z1 = np.dot(X, self.W1) + self.b1
        # A1 = Sinal de saída da camada oculta após passar pela função de ativação
        self.A1 = self.sigmoid(self.Z1)
        
        # Camada de Saída (Output Layer)
        # Z2 = Campo induzido da camada de saída (usa A1 como entrada)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # A2 = Saída final predita pela rede neural (y_hat)
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    def compute_loss(self, Y_hat, Y):
        """
        Função de Custo baseada no Erro Quadrático Médio (MSE) - Formulação de Simon Haykin.
        Mede o quão distante a predição da rede (Y_hat) está do valor real/desejado (Y).
        """
        # Y é o d(n) (valor desejado) e Y_hat é o y(n) (saída da rede)
        
        # Calcula o erro instantâneo e_j(n) = d_j(n) - y_j(n)
        erro = Y - Y_hat 
        
        # Calcula a energia do erro instantâneo E(n) para cada exemplo: 1/2 * soma(erro^2)
        E_n = 0.5 * np.sum(erro ** 2, axis=1) 
        
        # Calcula o erro quadrado médio E_av (média de todos os erros instantâneos)
        E_av = np.mean(E_n) 
        
        return E_av
    
    def backward(self, X, Y, learning_rate):
        """
        Processo de Backward Pass (Retropropagação do erro).
        Aplica a Regra da Cadeia para calcular os gradientes da função de perda 
        em relação a cada peso e bias, atualizando-os em seguida.
        """
        m = Y.shape[0] # Número de exemplos no lote (batch size)
        
        # --- CÁLCULOS PARA A CAMADA DE SAÍDA (W2, b2) ---
        # dZ2 representa o gradiente local (delta) da camada de saída.
        # Pela formulação do MSE, a derivada da perda em relação a Z2 é o erro multiplicado pela 
        # derivada da função de ativação local.
        # dZ2 = (A2 - Y) * phi'(Z2)
        dZ2 = (self.A2 - Y) * self.sigmoid_derivative(self.Z2) 
        
        # O gradiente dos pesos (dW2) é o produto escalar da entrada da camada (A1 transposta) pelo delta (dZ2).
        # Dividimos por 'm' para obter a média dos gradientes do lote.
        dW2 = np.dot(self.A1.T, dZ2) / m
        
        # O gradiente do bias (db2) é a soma das colunas de dZ2.
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        
        # --- CÁLCULOS PARA A CAMADA OCULTA (W1, b1) ---
        # Propagamos o erro dZ2 de volta para a camada oculta, multiplicando pelos pesos W2.
        dA1 = np.dot(dZ2, self.W2.T)
        
        # O gradiente local da camada oculta (dZ1) aplica a derivada da ativação nesta camada.
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
        
        # Calculamos os gradientes de W1 e b1 usando a entrada original (X) e o delta dZ1.
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        
        # --- ATUALIZAÇÃO DOS PESOS E BIASES ---
        # Aplicamos o Gradiente Descendente: movemos os parâmetros na direção oposta ao gradiente,
        # com o tamanho do passo controlado pela taxa de aprendizagem (learning_rate).
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def make_weights(self):
        """
        Gera pesos aleatórios para a rede, útil para testes ou reinicialização.
        """
        self.W1 = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
        print("Pesos e bias reinicializados com valores aleatórios.")

        return self.W1, self.b1, self.W2, self.b2

    def save_weights_files(self, filename):
        """
        Salva os pesos e bias da rede em um arquivo .npz compactado.
        """
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Pesos salvos em {filename}")

    def save_weights(self, W1, b1, W2, b2):
        """
        Salva os pesos e bias da rede em um arquivo .npz compactado.
        """
        np.savez("results/pesos_iniciais.npz", W1=W1, b1=b1, W2=W2, b2=b2)
        print(f"Pesos salvos em results/pesos_iniciais.npz")

    def load_weights(self, filename):
        """
        Carrega os pesos e bias da rede a partir de um arquivo .npz.
        """
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Pesos carregados de {filename}")