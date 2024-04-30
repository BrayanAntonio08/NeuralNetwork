import NeuralNetwork
import Neuron
import Layer
import numpy as np

print()
print('Bienvenido al sistema de entrenamiento de redes neuronales\n')

red = NeuralNetwork.NeuralNetwork()
red.readData('data/ex4data1.mat')
# red.readData('data/ex1.txt')

#cargar la configuracion
red.config(2,1,2)
red.build()

red.layers[0].loadValues('data/l1.txt')
red.layers[1].loadValues('data/l2.txt')

red.display()

print()
# n = Neuron.Neuron(2)
# n.genRandomValues()
# n.display()
# n.update([0.5, 0.2], 0.1)
# n.display()

# print(n.evaluate([1,2]))