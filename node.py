import numpy as np

class Node:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def activate(self, x):
        """
        Función de activación genérica. Debe ser implementada en las clases hijas.
        """
        raise NotImplementedError("Cada nodo debe tener su propia función de activación.")


    def forward(self, inputs):
        """
        Calcula la salida del nodo con la función de activación.
        """
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activate(weighted_sum)

     def __str__(self):
        """
        Representación en string del nodo.
        """
        return f"Node(input_size={self.num_inputs}, weights={self.weights}, bias={self.bias}
