import NeuralNetwork
import numpy as np

print()
print('Bienvenido al sistema de entrenamiento de redes neuronales\n')

red = NeuralNetwork.NeuralNetwork()
red.readData('data/ex4data1.mat')
# red.readData('data/ex1.txt')