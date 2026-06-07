class HiperparameterController:
    def __init__(self):
        self.input_size = None
        self.hidden_size = None
        self.output_size = None
        self.learning_rate = None
        self.max_epochs = None

    def set_hiperparameters(self, input_size, hidden_size, output_size, learning_rate, max_epochs):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def save_hiperparameters(self):
        with open("hiperparameter/hiperparametros.txt", "w") as f:
            f.write(f"Input Size: {self.input_size}\n")
            f.write(f"Hidden Size: {self.hidden_size}\n")
            f.write(f"Output Size: {self.output_size}\n")
            f.write(f"Learning Rate: {self.learning_rate}\n")
            f.write(f"Max Epochs: {self.max_epochs}\n")
    
    def get_hyperparameters(self):
        hidden_size = int(input("Digite o número de neurônios na camada escondida (ex: 15): ") or 15)
        learning_rate = float(input("Digite a taxa de aprendizado (ex: 0.1): ") or 0.1)
        max_epochs = int(input("Digite o número máximo de épocas (ex: 1000): ") or 1000)
        paciencia = int(input("Digite a paciência para Parada Antecipada (ex: 50): ") or 50)
        
        return hidden_size, learning_rate, max_epochs, paciencia