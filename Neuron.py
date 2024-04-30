import numpy as np
import random
class Neuron:
    def __init__(self, input_units = 0, values = np.array([])):
        self.weights = np.zeros(input_units)
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
            
        return result
    
    def update(self, deltas, alpha):
        '''
        Método de actualización de los valores (pesos) de la neurona. Recibe los deltas
        (derivadas parciales) por cada peso y el alpha que indica la taza de cambio
        '''
        for i in range(len(self.weights)):
            self.weights[i] -= alpha*deltas[i]
        return