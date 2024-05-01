import numpy as np
import random
class Neuron:
    def __init__(self, activation_function, input_units = 0, random_values=True, values = np.array([])):
        self.weights = np.zeros(input_units)
        self.activation_function = activation_function
        if(random_values):
            self.genRandomValues()
            
        if(len(values) > 0): self.weights = values
        pass
    
    def display(self):
        print(self.weights)
        
    def genRandomValues(self):
        for i in range(len(self.weights)):
            # se genera un valor aleatorio para cada peso entre 0.5 y -0.5
            self.weights[i] = random.uniform(-0.5, 0.5)
        return
    
    def evaluate(self, input_values):
        '''
        Este método se encarga de evaluar toda una fila de datos frente a los valores
        de la neurona, retorna un escalar
        '''
        result = 0
        for ivalue in range(len(self.weights)):
            result += input_values[ivalue]*self.weights[ivalue]
        
        self.net = result 
        return result
    
    def activate(self, value):
        activation = {
            'sigmoid': lambda: 1/(1+np.exp(-value))
        }
        self.out = activation[self.activation_function]()
        return self.out
    
    def activation_function_derivate(self, value):
        return
    
    def calculateDelta(self, back_delta):
        self.delta = back_delta * self.activation_function_derivate(self.net)
        return self.delta
    
    def update(self, alpha):
        '''
        Método de actualización de los valores (pesos) de la neurona. Recibe los deltas
        (derivadas parciales) por cada peso y el alpha que indica la taza de cambio
        '''
        for i in range(len(self.weights)):
            self.weights[i] -= alpha*deltas[i]
        return