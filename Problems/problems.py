import numpy as np
    
class Problems:
    def __init__(self, problems=None):
        self.problems = problems

    def __str__(self):
        count = len(self.problems) if self.problems else 0
        return f"ProblemsMPL with {count} problems"
    
    def load_from_csv(self, filepath, num_label_cols=1):
        """
        Lê um arquivo CSV genérico e separa os dados em atributos (X) e rótulos (Y).
        
        Parâmetros:
        - filepath: caminho/nome do arquivo CSV (ex: 'problemAND.csv')
        - num_label_cols: quantidade de colunas no final do arquivo que representam a saída desejada (Y).
        """
        # Carrega a matriz inteira do arquivo de texto, usando a vírgula como separador
        # IMPORTANTE: Se o seu CSV tiver um cabeçalho (nomes nas colunas), 
        # adicione o parâmetro 'skiprows=1' dentro do loadtxt.
        data = np.loadtxt(filepath, delimiter=',')
        
        # Fatiamento (Slicing) no Numpy: matriz[linhas, colunas]
        
        # X recebe todas as linhas (:), e todas as colunas do início até o limite do rótulo (:-num_label_cols)
        X = data[:, :-num_label_cols]
        
        # Y recebe todas as linhas (:), e apenas as colunas do limite do rótulo até o final (-num_label_cols:)
        Y = data[:, -num_label_cols:]
        
        return X, Y

    def XOR_problem(self):
        # Chama a função genérica informando que há apenas 1 coluna de rótulo no final
        return self.load_from_csv('problemXOR.csv', num_label_cols=1)
    
    def AND_problem(self):
        return self.load_from_csv('problemAND.csv', num_label_cols=1)
    
    def OR_problem(self):
        return self.load_from_csv('problemOR.csv', num_label_cols=1)
    
    def CARACTERES_COMPLETO_problem(self):
        # Carregando os dados de treinamento (versão limpa)
        # O np.load lê diretamente os arquivos binários do NumPy
        X_treino = np.load('X.npy')
        Y_treino = np.load('Y_classe.npy')
        
        # NOTA: Quando você baixar os arquivos com ruído para os testes, 
        # você pode carregá-los de forma idêntica:
        # X_teste = np.load('X_ruido.npy')
        # Y_teste = np.load('Y_classe_ruido.npy')
        
        return X_treino, Y_treino