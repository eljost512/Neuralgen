import numpy as np
from node import Node

class Perceptron(Node):
    def __init__(self, num_inputs):
        """
        Perceptrón con función de activación escalón.
        """
        super().__init__(num_inputs, activation_function=self.step_function, activation_derivative=self.step_derivative)

    @staticmethod
    def step_function(x):
        """
        Función escalón: devuelve 1 si x >= 0, 0 en caso contrario.
        """
        return 1 if x >= 0 else 0

    @staticmethod
    def step_derivative(x):
        """
        Derivada de la función escalón (no es útil para entrenamiento con gradiente).
        """
        return 0  # La derivada es 0 en todos los puntos excepto en x = 0 (donde es indefinida)

    def __str__(self):
        """
        Representación en string del perceptrón.
        """
        return f"Perceptron(inputs={self.num_inputs}, weights={self.weights}, bias={self.bias})"
