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

    def config(self, output_units, hidden_layers=0, hidden_units=0):
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        return
    
    def trainingConfig(self, data_filename, init_mode="random", weights_filenames=(), activation_function='sigmoid'):
        self.datafile = data_filename
        self.init_mode = init_mode
        self.weights_files = weights_filenames
        return
        
    def display(self):
        iLayer = 1
        for layer in self.layers:
            print('\n\tCapa %d'%iLayer)
            layer.display()
            iLayer += 1
        return
            
    def build(self):
        '''
        En este método se construiran las neuronas iniciales, divididas
        por capas y con valores aleatorios, en ceros, o valores definidos según un 
        archivo.
        
        Se valora un objeto [Layer] que contenga la lista de objetos [Neuron]
        
        Para el tipo de inicialización se utiliza la variable [init_mode] que debe
        tener un valor entre los siguientes: [random, zeros, loaded]
        '''
        units = self.hidden_units
        inputs = self.input_units + 1
        # la inicialización de los valores cambia según la configuracion
        initialization = {
            'random': lambda layer: 
                self.layers.append(Layer.Layer(units, inputs, random=True)),
            'zeros': lambda layer: 
                self.layers.append(Layer.Layer(units, inputs)),
            'loaded': lambda layer:
                self.layers.append(Layer.Layer(units, inputs, loadFile=self.weights_files[layer]))
        }
        
        for i_layer in range(self.hidden_layers):
            #crear una nueva capa
            initialization[self.init_mode](i_layer)
            inputs = self.hidden_units+1
            
        #crear la capa de salida
        units = self.output_units
        initialization[self.init_mode](self.hidden_layers)
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
        print('\nCargando información de entrenamiento...\n')
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
    
    def computeCost(self):
        return
    
    def train(self, maxiter=0):
        '''
        Acá se implementará el algoritmo de gradiente descendiente.
        Se comienza leyendo los datos preconfigurados y construyendo las neuronas iniciales.
        '''
        self.readData(self.datafile)
        #construir las capas iniciales de los valores
        self.build()
        
        #A partir de este punto se comienza un ciclo de evaluación de costos y optimización
        cost = self.computeCost()
        iteration = 0
        while True:
            #
            
            new_cost = self.computeCost()
            #evaluar diferencia de costos o posible divergencia
            if (cost-new_cost < 1e-6):
                print('Finalizando optimización por grado de presición')
                break
            if(maxiter>0 and iteration == maxiter):
                print('Se ha alcanzado el límite de iteraciones')
                break
        return