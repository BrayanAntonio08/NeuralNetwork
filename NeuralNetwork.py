import scipy.io as reader
import numpy as np
import Layer 
class NeuralNetwork:

    #constructor
    def __init__(self) -> None:
        # Datos de configuración
        self.input_units = 0
        self.output_units = 0
        self.hidden_layers = 0
        self.hidden_units = 0

        # variables para las operaciones
        self.training_samples = 0
        self.training_results = np.array([])
        self.training_set = np.array([])
        self.layers = []
        pass

    def config(self, output_units, hidden_layers = 0, hidden_units = 0):
        self.output_units = output_units
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        return
    
    def build(self):
        '''
        En este método se construiran las neuronas iniciales, divididas
        por capas y con valores aleatorios.
        
        Se valora un objeto [Layer] que contenga la lista de objetos [Neuron]
        '''
        input = self.input_units
        for hl in range(self.hidden_layers):
            #crear una nueva capa
            self.layers.append(Layer.Layer(self.hidden_units, input))
            input = self.hidden_units+1
            
        #crear la capa de salida
        self.layers.append(Layer.Layer(self.output_units, input))
        return
    
    def readData(self, filename):
        '''
        La lectura de datos puede ser a un txt dividido por comas o un archivo .mat.
        En ambos casos, se formará la matriz y se guardarán los datos en el objeto.
        
        Usa [training_set] para la matriz de X y [training_results] para los valores de Y.
        Adicionalmente se guardan la cantidad de entradas en [training_samples] y 
        de pesos o parámetros en [input_units]. Estos valores ayudarán a construir las
        matrices de valores para las neuronas
        '''
        print('\nCargando información de entrenamiento...')
        if(filename.endswith('.mat')):
            data = reader.loadmat(filename)
            self.training_set, self.training_results = data['X'], data['y']
            
            #se define la cantidad de datos de entrada según el archivo
            self.training_samples = len(self.training_set)
            self.input_units = len(self.training_set[0])
            
            print("Datos cargados en mat")
            print("Cantidad de muestras: %d \nCantidad de parámetros: %d"%(self.training_samples, self.input_units)) 
            
        elif(filename.endswith('.txt')):
            data = np.loadtxt(filename, delimiter=',')
            if(data.ndim == 1):
                data = data.reshape(1, len(data))
            
            self.training_set = np.array(data[:,:-1])
            self.training_results = np.array(data[:,-1:])
            
            #se define la cantidad de datos de entrada según el archivo
            self.training_samples = len(self.training_set)
            self.input_units = len(self.training_set[0])
            
            print("Datos cargados en txt")
            print("Cantidad de muestras: %d \nCantidad de parámetros: %d"%(self.training_samples, self.input_units))
        else:
            print('No se ha definido un tipo de archivo valido')
            
        return 
    
    def train(self):
        return