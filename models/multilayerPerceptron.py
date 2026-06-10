import numpy as np

class MultilayerPerceptron:
    """
    Implementação de uma Rede Neural Artificial Multilayer Perceptron (MLP) com uma camada oculta.
    Treinamento realizado através do algoritmo Backpropagation (Gradiente Descendente em Lote/Batch).
    Esta estrutura permite o aprendizado de funções não-lineares complexas através de ajustes iterativos.
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Definição da arquitetura da rede (Hiperparâmetros)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicialização de Pesos (W) e Viéses (b - bias)
        # Utilizamos valores aleatórios pequenos no intervalo [-1, 1] para quebrar a simetria inicial,
        # garantindo que cada neurônio aprenda características diferentes durante a fase de treino,
        # evitando o problema onde neurônios em uma mesma camada convergem para os mesmos valores.
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        
        # O bias é inicializado com zeros para cada neurônio da camada. Sua função é permitir que a função
        # de ativação seja deslocada para a esquerda ou direita, proporcionando flexibilidade ao modelo
        # para ajustar o limiar de ativação e melhor se adaptar aos dados de entrada fornecidos.
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        """
        Função de ativação Sigmoide. 
        Mapeia qualquer valor real para um intervalo entre 0 e 1, introduzindo não-linearidade na rede.
        Sem esta função, a rede se comportaria como uma regressão linear simples, independentemente da 
        quantidade de camadas ou neurônios ocultos, sendo incapaz de aprender padrões complexos.
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Derivada da função de ativação Sigmoide.
        Esta função é essencial para o algoritmo de Backpropagation, onde calculamos o gradiente
        da função de custo. Ela indica a sensibilidade da saída em relação a pequenas mudanças 
        nos pesos, permitindo saber a direção e magnitude dos ajustes necessários via Regra da Cadeia.
        """
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def feed_forward(self, X):
        """
        Processo de Forward Pass (Propagação para frente).
        Calcula a saída da rede neural passando os dados de entrada camada por camada.
        O sinal percorre a rede sofrendo transformações lineares (dot product) e não-lineares 
        (ativação), resultando na predição final (Y_hat) que será comparada com o alvo real.
        """
        # Camada Oculta: Z1 armazena a soma ponderada das entradas + bias (campo induzido).
        # A1 é o sinal resultante após a função de ativação, que serve como entrada para a próxima camada.
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        
        # Camada de Saída: Z2 processa os sinais da camada oculta (A1) usando os pesos W2 e bias b2.
        # A2 representa a saída final da rede (y_hat), que passará pela função de perda para 
        # avaliarmos a acurácia do modelo em relação aos dados reais fornecidos.
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    def compute_loss(self, Y_hat, Y):
        """
        Função de Custo baseada no Erro Quadrático Médio (MSE).
        Mede a discrepância entre a predição da rede (Y_hat) e o valor desejado (Y).
        O erro é elevado ao quadrado para penalizar erros maiores mais severamente, e a média dos 
        erros ao longo do batch é utilizada para guiar a otimização dos pesos da rede.
        """
        # O erro instantâneo e_j é a diferença entre o valor desejado e a saída da rede.
        # Elevamos ao quadrado para garantir que o erro seja sempre positivo e diferenciável,
        # facilitando a aplicação do cálculo diferencial para minimizar o erro na rede.
        erro = Y - Y_hat 
        
        # A energia do erro E(n) para o exemplo é a soma dos quadrados dos erros de todos os neurônios.
        # Multiplicamos por 0.5 para que, ao derivar, o expoente 2 seja cancelado, simplificando
        # o cálculo do gradiente matemático que será utilizado na atualização dos pesos.
        E_n = 0.5 * np.sum(erro ** 2, axis=1) 
        
        # O erro médio (E_av) é a média global da energia de erro calculada sobre todo o lote (batch).
        # Este valor representa a "temperatura" do erro total do sistema no momento da iteração.
        E_av = np.mean(E_n) 
        
        return E_av
    
    def backward(self, X, Y, learning_rate):
        """
        Processo de Backward Pass (Retropropagação do erro).
        Utiliza a Regra da Cadeia para derivar a função de custo em relação aos pesos e bias.
        O erro é propagado da camada de saída até a de entrada, permitindo ajustes precisos em 
        cada conexão para reduzir o erro global na próxima iteração do ciclo de aprendizado.
        """
        m = Y.shape[0] # Número de exemplos no lote (batch size)
        
        # --- CÁLCULOS PARA A CAMADA DE SAÍDA (W2, b2) ---
        # dZ2 é o gradiente local (delta) que representa o erro da saída ponderado pela sensibilidade
        # da função de ativação. Ele é a chave para o ajuste: se a ativação está longe do ideal,
        # dZ2 será grande e causará uma alteração maior nos pesos da camada.
        dZ2 = (self.A2 - Y) * self.sigmoid_derivative(self.Z2) 
        
        # dW2 computa como cada peso contribuiu para o erro total da rede através da entrada da camada.
        # Dividimos por 'm' para obter o gradiente médio do lote, o que garante que a atualização
        # dos pesos seja estável e representativa de todo o conjunto de dados processado.
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # --- CÁLCULOS PARA A CAMADA OCULTA (W1, b1) ---
        # dA1 retropropaga o erro da saída para a camada oculta via matriz de pesos W2 transposta.
        # É a Regra da Cadeia em ação: descobrimos qual fração do erro final foi responsabilidade
        # de cada neurônio na camada oculta, baseando-se na força das conexões W2.
        dA1 = np.dot(dZ2, self.W2.T)
        
        # dZ1 aplica a derivada da ativação local (Z1) para obter a influência real no erro.
        # Com esses gradientes, calculamos como W1 e b1 devem ser modificados para que, no próximo
        # ciclo, o erro da camada oculta (e consequentemente o da rede) seja menor.
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # --- ATUALIZAÇÃO DOS PESOS E BIASES ---
        # Aplicamos a regra do Gradiente Descendente: subtraímos uma fração (learning_rate) do
        # gradiente atual. Isso desloca os parâmetros na direção que minimiza a função de custo,
        # melhorando o desempenho preditivo da rede após cada rodada de treinamento.
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def make_weights(self):
        """
        Gera pesos aleatórios para a rede.
        Reseta os pesos e bias da rede neural, permitindo reiniciar o experimento do zero.
        Isso é fundamental para realizar análises de robustez, comparando como diferentes 
        inicializações aleatórias afetam a convergência e o desempenho final do modelo.
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
        Armazena o estado atual da rede (pesos e bias) em formato binário.
        Isso permite persistir o aprendizado do modelo, garantindo que o conhecimento adquirido 
        durante o treinamento possa ser carregado posteriormente sem a necessidade de re-treino.
        """
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Pesos salvos em {filename}")

    def save_weights(self, W1, b1, W2, b2):
        """
        Salva uma cópia dos pesos iniciais em um arquivo fixo.
        Esta função serve para registrar o estado inicial da rede, permitindo o rastreamento 
        do histórico de mudanças. Essencial para verificar se o treinamento foi estável e para
        auditoria técnica caso o comportamento da rede pareça anômalo durante o processo.
        """
        np.savez("results/pesos_iniciais.npz", W1=W1, b1=b1, W2=W2, b2=b2)
        print(f"Pesos salvos em results/pesos_iniciais.npz")

    def load_weights(self, filename):
        """
        Carrega os pesos e bias da rede a partir de um arquivo .npz.
        Recupera os parâmetros treinados, substituindo os pesos atuais da instância.
        Com essa função, podemos implantar o modelo em produção ou continuar o treinamento 
        de onde paramos, garantindo que a rede mantenha o "conhecimento" aprendido anteriormente.
        """
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Pesos carregados de {filename}")