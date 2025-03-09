import numpy as np
import warnings

class optimization():

    def update_gradients(fashionnet, grad_parameters, wandb_configs, m, v, t):
        warnings.filterwarnings("error")
        if wandb_configs.optimizer == "sgd":
            for i in range(1, fashionnet.total_layers):
                index_w = "W" + str(i)
                index_b = "b" + str(i)
                fashionnet.parameters[index_w] = fashionnet.parameters[index_w] - wandb_configs.learning_rate * grad_parameters[index_w]
                fashionnet.parameters[index_b] = fashionnet.parameters[index_b] - wandb_configs.learning_rate * grad_parameters[index_b]
            
        elif wandb_configs.optimizer == "momentum":
            for i in range(1, fashionnet.total_layers):
                index_w = "W" + str(i)
                index_b = "b" + str(i)
                m[index_w] = wandb_configs.momentum * m[index_w] + (1 - wandb_configs.momentum) * grad_parameters[index_w]
                m[index_b] = wandb_configs.momentum * m[index_b] + (1 - wandb_configs.momentum) * grad_parameters[index_b]
                fashionnet.parameters[index_w] = fashionnet.parameters[index_w] - wandb_configs.learning_rate * m[index_w]
                fashionnet.parameters[index_b] = fashionnet.parameters[index_b] - wandb_configs.learning_rate * m[index_b]
        
        elif wandb_configs.optimizer == "nag":
            for i in range(1, fashionnet.total_layers):
                index_w = "W" + str(i)
                index_b = "b" + str(i)
                m[index_w] = wandb_configs.momentum * m[index_w] + wandb_configs.learning_rate * grad_parameters[index_w]
                m[index_b] = wandb_configs.momentum * m[index_b] + wandb_configs.learning_rate * grad_parameters[index_b]
                fashionnet.parameters[index_w] = fashionnet.parameters[index_w] - m[index_w]
                fashionnet.parameters[index_b] = fashionnet.parameters[index_b] - m[index_b]


        elif wandb_configs.optimizer == "rmsprop":

            for i in range(1, fashionnet.total_layers):
                index_w = "W" + str(i)
                index_b = "b" + str(i)
                v[index_w] = wandb_configs.beta * v[index_w] + (1 - wandb_configs.beta) * (grad_parameters[index_w] ** 2)
                v[index_b] = wandb_configs.beta * v[index_b] + (1 - wandb_configs.beta) * (grad_parameters[index_b] ** 2)
                fashionnet.parameters[index_w] = fashionnet.parameters[index_w] - (wandb_configs.learning_rate * grad_parameters[index_w]) / (np.sqrt(v[index_w]) + wandb_configs.epsilon)
                fashionnet.parameters[index_b] = fashionnet.parameters[index_b] - (wandb_configs.learning_rate * grad_parameters[index_b]) / (np.sqrt(v[index_b]) + wandb_configs.epsilon) 

        elif wandb_configs.optimizer == "adam":

            for i in range(1, fashionnet.total_layers):
                index_w = "W" + str(i)
                index_b = "b" + str(i)
                m[index_w] = wandb_configs.beta1 * m[index_w] + (1 - wandb_configs.beta1) * grad_parameters[index_w]
                m[index_b] = wandb_configs.beta1 * m[index_b] + (1 - wandb_configs.beta1) * grad_parameters[index_b]
                v[index_w] = wandb_configs.beta2 * v[index_w] + (1 - wandb_configs.beta2) * (grad_parameters[index_w] ** 2)
                v[index_b] = wandb_configs.beta2 * v[index_b] + (1 - wandb_configs.beta2) * (grad_parameters[index_b] ** 2)
                mhw = m[index_w] / (1 - np.power(wandb_configs.beta1,t))
                mhb = m[index_b] / (1 - np.power(wandb_configs.beta1,t))
                vhw = v[index_w] / (1 - np.power(wandb_configs.beta2,t))
                vhb = v[index_b] / (1 - np.power(wandb_configs.beta2,t))               
                fashionnet.parameters[index_w] = fashionnet.parameters[index_w] - (wandb_configs.learning_rate * mhw) / (np.sqrt(vhw) + wandb_configs.epsilon)
                fashionnet.parameters[index_b] = fashionnet.parameters[index_b] - (wandb_configs.learning_rate * mhb) / (np.sqrt(vhb) + wandb_configs.epsilon)            


        elif wandb_configs.optimizer == "nadam":

            for i in range(1, fashionnet.total_layers):
                index_w = "W" + str(i)
                index_b = "b" + str(i)
                m[index_w] = wandb_configs.beta1 * m[index_w] + (1 - wandb_configs.beta1) * grad_parameters[index_w]
                m[index_b] = wandb_configs.beta1 * m[index_b] + (1 - wandb_configs.beta1) * grad_parameters[index_b]
                v[index_w] = wandb_configs.beta2 * v[index_w] + (1 - wandb_configs.beta2) * (grad_parameters[index_w] ** 2)
                v[index_b] = wandb_configs.beta2 * v[index_b] + (1 - wandb_configs.beta2) * (grad_parameters[index_b] ** 2)
                mhw = m[index_w] / (1 - np.power(wandb_configs.beta1,t))
                mhb = m[index_b] / (1 - np.power(wandb_configs.beta1,t))
                vhw = v[index_w] / (1 - np.power(wandb_configs.beta2,t))
                vhb = v[index_b] / (1 - np.power(wandb_configs.beta2,t))                

                #print(vhw)
                #print(np.sqrt(vhw))
                	
                fashionnet.parameters[index_w] = fashionnet.parameters[index_w] - (wandb_configs.learning_rate / (np.sqrt(vhw) + wandb_configs.epsilon)) * ((wandb_configs.beta1 * mhw) + (((1 - wandb_configs.beta1) * grad_parameters[index_w]) / (1 - wandb_configs.beta1 ** t)))
                fashionnet.parameters[index_b] = fashionnet.parameters[index_b] - (wandb_configs.learning_rate / (np.sqrt(vhb) + wandb_configs.epsilon)) * ((wandb_configs.beta1 * mhb) + (((1 - wandb_configs.beta1) * grad_parameters[index_b]) / (1 - wandb_configs.beta1 ** t)))        

        else:
            raise ValueError("Invalid optimizer")
        
        t = t + 1
        
        return m, v, t
    
    def initialize_m_v(fashionnet):

        m = dict()
        v = dict()

        for i in range(1, fashionnet.total_layers):

            index_w = "W" + str(i)
            index_b = "b" + str(i)

            m[index_w] = np.zeros(fashionnet.parameters[index_w].shape)
            m[index_b] = np.zeros(fashionnet.parameters[index_b].shape)
            v[index_w] = np.zeros(fashionnet.parameters[index_w].shape)
            v[index_b] = np.zeros(fashionnet.parameters[index_b].shape)
        
        return m, v


