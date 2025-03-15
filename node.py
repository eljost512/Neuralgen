import numpy as np

class Node:
    def __init__(self, num_inputs, activation_function=None, activation_derivative=None):
        """
        Clase base para una neurona genérica.
        :param num_inputs: Número de entradas a la neurona.
        :param activation_function: Función de activación.
        :param activation_derivative: Derivada de la función de activación.
        """
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)  # Pesos inicializados aleatoriamente
        self.bias = np.random.randn()  # Sesgo inicializado aleatoriamente
        self.activation_function = activation_function if activation_function else self.relu
        self.activation_derivative = activation_derivative if activation_derivative else self.relu_derivative

    def forward(self, inputs):
        """
        Realiza la propagación de la señal.
        :param inputs: Valores de entrada.
        :return: Salida activada de la neurona.
        """
        self.inputs = np.array(inputs)
        self.z = np.dot(self.inputs, self.weights) + self.bias  # Suma ponderada
        self.output = self.activation_function(self.z)
        return self.output

    def backward(self, error, learning_rate):
        """
        Aplica retropropagación para actualizar los pesos.
        :param error: Gradiente del error con respecto a la salida.
        :param learning_rate: Tasa de aprendizaje.
        """
        delta = error * self.activation_derivative(self.output)
        self.weights -= learning_rate * delta * self.inputs
        self.bias -= learning_rate * delta

    def reset(self):
        """
        Reinicia los pesos y el bias aleatoriamente.
        """
        self.weights = np.random.randn(self.num_inputs)
        self.bias = np.random.randn()

    def __str__(self):
        """
        Representación en string de la neurona.
        """
        return f"Neuron(inputs={self.num_inputs}, weights={self.weights}, bias={self.bias})"

    # Funciones de activación predefinidas
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return Node.sigmoid(x) * (1 - Node.sigmoid(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
