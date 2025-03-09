import numpy as np

#Creating the Neural Network Class

class NN:

  '''The NN class creates the neural network. It takes 4 inputs.
  *input_neurons -  This is the number of neurons in the input layer. Typically this is the number of features in the input data.
  *In our application, it is the number of pixels in a fashion MNIST image.
  *hidden_layers - This is the number of hidden layers we want in our dataset. The default value is 1
  *hidden_neurons - This is the number of neurons in the hidden layer'''

  def __init__(self, input_neurons, output_neurons, wandb_configs):

    #Structure of Neural Network

    self.input_neurons = input_neurons
    self.output_neurons = output_neurons
    self.hidden_neurons = wandb_configs.hidden_size
    self.num_layers = wandb_configs.num_layers
    self.total_layers = self.num_layers + 2

    dimensions = [self.input_neurons] + [self.hidden_neurons]*self.num_layers + [self.output_neurons]
    self.parameters = dict()

    #Initialization of Weights and Biases
    for i in range(1, self.total_layers):
        
        np.random.seed(42)
        index_w = "W" + str(i)
        index_b = "b" + str(i)
        #Assuming Normal distribution [0,1) for both cases:
        if wandb_configs.weight_init == "random":
            self.parameters[index_w] = np.random.randn(dimensions[i], dimensions[i-1]) * 0.05
        elif wandb_configs.weight_init == "Xavier":
            limit = np.sqrt(2/(dimensions[i] + dimensions[i-1]))
            self.parameters[index_w]= np.random.randn(dimensions[i],dimensions[i-1]) * limit
        else:
           raise ValueError("Please enter only random or Xavier")

        self.parameters[index_b] = np.zeros((dimensions[i], 1))
       

    self.activations = dict()



