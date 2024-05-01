import Neuron
import numpy as np
class Layer:
    def __init__(self, units, input_units, random=False, loadFile=''):
        self.units = units
        self.input_units = input_units
        self.neurons = []
        
        for i in range(units):
            neuron = Neuron.Neuron(input_units, random)
            self.neurons.append(neuron)
        
        if(len(loadFile)>0):
            self.loadValues(loadFile)
        pass
    
    def loadValues(self, filename):
        data = np.loadtxt(filename, delimiter=',')
        for i in range(self.units):
            self.neurons[i].weights = data[i]
            
    def display(self):
        for neuron in self.neurons:
            neuron.display()
            
    def evaluate(self, entry):
        
        return