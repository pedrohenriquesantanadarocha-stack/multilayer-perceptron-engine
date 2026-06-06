from training import Training
from multilayerPerceptron import MultilayerPerceptron
from problems import Problems

# Example usage
if __name__ == "__main__":
    # Create a Multilayer Perceptron model
    input_size = 2
    hidden_size = 2
    output_size = 1
    model = MultilayerPerceptron(input_size, hidden_size, output_size)
    
    # Create a Training instance
    trainer = Training(model, learning_rate=0.01, epochs=10000)
    
    # Define the XOR problem dataset
    problems = Problems(problems=[])
    x, y = problems.XOR_problem()
    
    # Train the model on the XOR problem
    trainer.fit(x, y)
    
    # Test the trained model
    predictions = model.forward(x)
    print("Predictions:")
    print(predictions)
    