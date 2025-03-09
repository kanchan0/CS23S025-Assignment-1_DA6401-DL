import numpy as np

class activations():
    """
    This class compute the activations and the gradient of activations
    
    """
    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        # Avoid numerical instability by subtracting the maximum value
        exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    @staticmethod
    def grad_identity(x):
        return np.ones_like(x)  # Gradient of identity function is always 1

    @staticmethod
    def grad_sigmoid(x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)  # Gradient of sigmoid function

    @staticmethod
    def grad_tanh(x):
        th = np.tanh(x)
        return 1 - th**2  # Gradient of tanh function

    @staticmethod
    def grad_relu(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def grad_softmax(x):
        exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
        smax = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        return smax * (1 - smax)
    

    def activate(input_matrix, activation_function):

        """This function computes activations"""

        if activation_function == "identity":
            return activations.identity(input_matrix)
        elif activation_function == "sigmoid":
            return activations.sigmoid(input_matrix)
        elif activation_function == "tanh":
            return activations.tanh(input_matrix)
        elif activation_function == "ReLU":
            return activations.relu(input_matrix)
        elif activation_function == "softmax":
            return activations.softmax(input_matrix)
        else:
            raise ValueError("Invalid activation function")
        
    def grad_activate(input_matrix, activation_function):

        """This function computes gradient of activations"""
        if activation_function == "identity":
            return activations.grad_identity(input_matrix)
        elif activation_function == "sigmoid":
            return activations.grad_sigmoid(input_matrix)
        elif activation_function == "tanh":
            return activations.grad_tanh(input_matrix)
        elif activation_function == "ReLU":
            return activations.grad_relu(input_matrix)
        elif activation_function == "softmax":
            return activations.grad_softmax(input_matrix)
        else:
            raise ValueError("Invalid activation function")


class loss():

    def compute_loss(predictions, labels, loss_function, batch_size, weight_decay, fashionnet):

        if loss_function == "cross_entropy":

            losses = -1 * np.sum(np.multiply(labels, np.log(predictions))) / batch_size

        elif loss_function == "mean_squared_error":

            losses = (1/2) * np.sum((predictions - labels)**2) / batch_size

        else:

            raise ValueError("Invalid Loss Function")
        
        #Adding the regularization term to the loss functions

        #norm = 0

        #for i in range(1, fashionnet.total_layers):

        #    norm += np.power(fashionnet.parameters["W" + str(i)], 2)

        #losses = losses + (weight_decay/(2*batch_size)) * norm

        return losses


        
    def compute_grad_loss(predictions, label_train, loss_function, a):

        #Computes the gradient of the loss functions 
        
        if loss_function == "cross_entropy":

            gradout = -(label_train - predictions)

        elif loss_function == "mean_squared_error":
            
            gradout = np.multiply(2 *(predictions - label_train), activations.grad_activate(a, "softmax"))
        
        return gradout


    def calculate_accuracy(predictions, actual_labels):
     
        predicted_labels = np.argmax(predictions, axis=0)
        
        correct_predictions = np.sum(predicted_labels == actual_labels)
        
        total_samples = len(actual_labels)
        
        accuracy = correct_predictions / total_samples * 100
        
        return accuracy
