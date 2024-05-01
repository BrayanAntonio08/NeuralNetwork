import NeuralNetwork
import Neuron
import Layer
import numpy as np

print()
print('Bienvenido al sistema de entrenamiento de redes neuronales\n')

red = NeuralNetwork.NeuralNetwork()
# red.readData('data/ex4data1.mat')

#cargar la configuracion
red.config(\
    output_units=2,\
    hidden_layers=1,\
    hidden_units=2)

#configurar el modo de entrenamiento
red.trainingConfig(\
    data_filename='data/ex1.txt',\
    init_mode='loaded',\
    weights_filenames=['data/l1.txt', 'data/l2.txt'])

red.train()


red.display()

print()
# n = Neuron.Neuron(2)
# n.genRandomValues()
# n.display()
# n.update([0.5, 0.2], 0.1)
# n.display()

# print(n.evaluate([1,2]))