import numpy as np 

class Problems:
    """
    Classe utilitária para o carregamento e preparação de datasets para treinamento de redes neurais.
    Esta estrutura centraliza a lógica de leitura de arquivos (CSV ou binários) e a separação 
    dos dados em matrizes de entrada (features) e saída (labels), facilitando a integração 
    dos problemas com o modelo MLP.
    """
    def __init__(self, problems=None):
        self.problems = problems

    def __str__(self):
        count = len(self.problems) if self.problems else 0
        return f"ProblemsMPL com {count} problemas carregados."
    
    def load_from_csv(self, filepath, num_label_cols=1):
        """
        Lê um arquivo CSV genérico e separa os dados em atributos (X) e rótulos (Y).
        Esta função é a base para ler datasets tabulares, realizando o fatiamento correto
        das colunas para isolar o que a rede deve aprender (X) do que ela deve predizer (Y).
        
        Parâmetros:
        - filepath: caminho/nome do arquivo CSV (ex: 'problemAND.csv')
        - num_label_cols: quantidade de colunas no final que representam a saída desejada (Y).
        """
        # Carrega a matriz inteira do arquivo, utilizando a vírgula como delimitador.
        # O encoding 'utf-8-sig' é utilizado para lidar com caracteres especiais, 
        # comum em arquivos CSV gerados por planilhas como Excel.
        data = np.loadtxt(filepath, delimiter=',', encoding='utf-8-sig')
        
        # Fatiamento (Slicing) no NumPy: matriz[linhas, colunas]
        # X recebe todas as linhas (:), e todas as colunas do início até o limite dos rótulos.
        # A notação [:, :-num_label_cols] exclui as últimas 'n' colunas de saída.
        X = data[:, :-num_label_cols]
        
        # Y recebe todas as linhas (:), e apenas as colunas do limite do rótulo até o final.
        # A notação [:, -num_label_cols:] extrai especificamente as colunas de alvo.
        Y = data[:, -num_label_cols:]
        
        return X, Y

    def XOR_problem(self):
        """
        Carrega o dataset clássico do problema XOR (Ou Exclusivo).
        Este problema é não-linearmente separável, sendo o caso de teste fundamental 
        para provar a necessidade de uma camada oculta na arquitetura MLP.
        """
        return self.load_from_csv('problems/ProblemsExamples/problemXOR.csv', num_label_cols=1)
    
    def AND_problem(self):
        """
        Carrega o dataset para a porta lógica AND.
        Um problema linearmente separável simples, utilizado como teste de sanity check
        para validar se o algoritmo de retropropagação está convergindo corretamente.
        """
        return self.load_from_csv('problems/ProblemsExamples/problemAND.csv', num_label_cols=1)
    
    def OR_problem(self):
        """
        Carrega o dataset para a porta lógica OR.
        Assim como a porta AND, é um problema linearmente separável, servindo como base
        para verificar a capacidade da rede de aprender regras de classificação binária.
        """
        return self.load_from_csv('problems/ProblemsExamples/problemOR.csv', num_label_cols=1)
    
    def CARACTERES_COMPLETO_problem(self):
        """
        Carrega os dados de caracteres (matrizes de pixels) em formato binário NumPy
        """
        # Carregando os dados de treinamento (versão limpa) diretamente do disco.
        X_treino = np.load('problems/ProblemsExamples/X_classe.npy')
        Y_treino = np.load('problems/ProblemsExamples/Y_classe.npy')
        
        # Estratégia de dados ruidosos: ao utilizar datasets com ruído, a rede aprende 
        # a ser mais robusta, realizando uma generalização melhor mesmo quando os
        # dados de entrada possuem variações ou pequenas corrupções nos pixels.
        return X_treino, Y_treino