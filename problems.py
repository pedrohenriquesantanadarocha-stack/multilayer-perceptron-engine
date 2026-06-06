import numpy as np
    
class Problems:
    def __init__(self, problems):
        self.problems = problems

    def __str__(self):
        return f"ProblemsMPL with {len(self.problems)} problems"
    
    def XOR_problem(self):
        return np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])
    
    def AND_problem(self):
        return np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [0], [0], [1]])
    
    def OR_problem(self):
        return np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [1]])
    
    def CARACTER_problem(self):
        # This is a placeholder for a more complex problem, such as character recognition
        # For simplicity, we will use a small dataset of binary patterns representing characters
        X = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        Y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])  # Example labels for characters
        return X, Y