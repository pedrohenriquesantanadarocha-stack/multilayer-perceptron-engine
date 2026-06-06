class Training:
    def __init__(self, model, learning_rate=0.01, epochs=1000):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, x, y):
        for epoch in range(self.epochs):
            for i in range(len(x)):
                # Forward pass
                X_batch = x[i:i+1]
                y_batch = y[i:i+1]
                
                output = self.model.forward(X_batch)
                
                # Backward pass and update weights
                self.model.backward(X_batch, y_batch, self.learning_rate)