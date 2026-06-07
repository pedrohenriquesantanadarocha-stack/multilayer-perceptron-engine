from problems.problems import Problems

class ControllerProblems:
    def __init__(self):
        self.problems = Problems()
    
    def select_problem_interactively(self):
        print("Select a problem to train the model:")
        print("1. XOR")
        print("2. AND")
        print("3. OR")
        print("4. CARACTERES_COMPLETO")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            x, y = self.problems.XOR_problem()
        elif choice == '2':
            x, y = self.problems.AND_problem()

        elif choice == '3':
            x, y = self.problems.OR_problem()
        elif choice == '4':
            x, y = self.problems.CARACTERES_COMPLETO_problem()
        elif choice == '5':
            print("Exiting the program.")
            exit()
        else:
            print("Invalid choice. Please try again.")

        return x, y