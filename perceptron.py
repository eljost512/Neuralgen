from node import Node

class Perceptron(Node):
    def __init__(self, num_inputs):
        super().__init__(num_inputs) #Constrctor de la clase heredada
    
    def activate(self, x):
        """
        Función escalón del perceptrón.
        """
        return 1 if x >= 0 else 0