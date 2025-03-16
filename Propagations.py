import numpy as np
from ActivationdANDLoss import activations
from ActivationdANDLoss import loss

class propagations():
    """
    This class contains two methods which implements forward propagation and backward propagation
    """

    def forward_propagation(fashionnet, img_train, wandb_configs):


        for i in range(1, fashionnet.total_layers):

            index_a = "a" + str(i)
            index_h = "h" + str(i)

            index_w = "W" + str(i)
            index_b = "b" + str(i)

            #print(fashionnet.parameters['b2'])

            if i == 1:
                fashionnet.activations[index_a] = np.matmul(fashionnet.parameters[index_w], img_train) + fashionnet.parameters[index_b]
            else:
                fashionnet.activations[index_a] = np.matmul(fashionnet.parameters[index_w], fashionnet.activations["h" + str(i-1)]) + fashionnet.parameters[index_b]

            if i == fashionnet.total_layers - 1:
                predictions = activations.activate(fashionnet.activations[index_a], wandb_configs.output_activation)
            else:
                fashionnet.activations[index_h] = activations.activate(fashionnet.activations[index_a], wandb_configs.activation)

        return predictions
    
    def backward_propagation(fashionnet, img_train, label_train, predictions, wandb_configs, start_index, end_index, h):

        grad_parameters = {}
        grad_activations = {}

        grad_activations["a" + str(fashionnet.total_layers-1)] = loss.compute_grad_loss(predictions, label_train, wandb_configs.loss, h) 
        
        for i in reversed(range(1, fashionnet.total_layers-1)):

            curr_h = "h" + str(i)
            #prev_h = "h" + str(i+1)

            curr_a = "a" + str(i)
            prev_a = "a" + str(i+1)

            grad_activations[curr_h] = np.matmul(fashionnet.parameters["W" + str(i+1)].T, grad_activations[prev_a])
            g_prime = activations.grad_activate(fashionnet.activations[curr_a], wandb_configs.activation)

            grad_activations[curr_a] = np.multiply(grad_activations[curr_h], g_prime)
        
        for i in range(1, fashionnet.total_layers):

            index_w = "W" + str(i)
            index_b = "b" + str(i)

            index_h = "h" + str(i-1)
            index_a = "a" + str(i)

            regularization = wandb_configs.weight_decay * fashionnet.parameters[index_w]

            if i == 1:

                grad_parameters[index_w] = (np.dot(grad_activations[index_a], img_train.T) + regularization) / (end_index - start_index + 1)
                
            else:

                grad_parameters[index_w] = (np.matmul(grad_activations[index_a], fashionnet.activations[index_h].T) + regularization) / (end_index - start_index + 1)

            grad_parameters[index_b] = np.sum(grad_activations[index_a], axis=1, keepdims=True)

            grad_parameters[index_w] = grad_parameters[index_w].astype(np.float64)
            grad_parameters[index_b] = grad_parameters[index_b].astype(np.float64)

            

        return grad_parameters


