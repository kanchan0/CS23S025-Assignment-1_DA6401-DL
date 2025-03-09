import numpy as np
import argparse
import wandb
from keras.datasets import fashion_mnist, mnist
from NeuralNetwork import NN
from Propagations import propagations
from Optimizor4BackProp import optimization
from ActivationdANDLoss import loss


import wandb

class wbhelp():

    def set_configs(args):

        wandb_configs = {

        'wandb_project'		:  args.wandb_project,
        'wandb_entity'      :  args.wandb_entity,
        'dataset'           :  args.dataset,
        'epochs'            :  args.epochs,
        'batch_size'        :  args.batch_size,
        'loss'              :  args.loss,
        'optimizer'         :  args.optimizer,
        'learning_rate'     :  args.learning_rate,
        'momentum'          :  args.momentum,
        'beta'              :  args.beta,
        'beta1'             :  args.beta1,
        'beta2'             :  args.beta2,
        'epsilon'           :  args.epsilon,
        'weight_decay'      :  args.weight_decay,
        'weight_init'       :  args.weight_init,
        'num_layers'        :  args.num_layers,
        'hidden_size'       :  args.hidden_size,
        'activation'        :  args.activation,
        'output_activation' :  args.output_activation,
        }

        return wandb_configs
    
    def run_name(wandb_configs):

        run_name = "fashion_mnist_nhl_{}_sz_{}_a_{}_o_{}_lr_{}_e_{}_b_{}_l_{}_wi_{}".format(
            wandb_configs["num_layers"],
            wandb_configs["hidden_size"],
            wandb_configs["activation"],
            wandb_configs["optimizer"],
            wandb_configs["learning_rate"],
            wandb_configs["epochs"],
            wandb_configs["batch_size"],
            wandb_configs["loss"],
            wandb_configs["weight_init"]
        )

        return run_name
    
    def sweep_name(wandb_configs):

        sweep_name = "fashion_mnist_nhl_{}_sz_{}_a_{}_o_{}_lr_{}_e_{}_b_{}_l_{}_wi_{}".format(
            wandb_configs.num_layers,
            wandb_configs.hidden_size,
            wandb_configs.activation,
            wandb_configs.optimizer,
            wandb_configs.learning_rate,
            wandb_configs.epochs,
            wandb_configs.batch_size,
            wandb_configs.loss,
            wandb_configs.weight_init
        )

        return sweep_name




def train(wandb_configs):


    if wandb_configs["dataset"] == "fashion_mnist":
        (img_train, lbl_train), (img_test, lbl_test) = fashion_mnist.load_data()

    elif wandb_configs["dataset"] == "mnist":
        (img_train, lbl_train), (img_test, lbl_test) = mnist.load_data()

    else:
        raise ValueError("Invalid dataset name")

    #Initialise wandb
    run_name = wbhelp.run_name(wandb_configs)
    wandb.init(config=wandb_configs, project="CS23S025-Assignment-1-DA6401-DL", entity="cs23s025-indian-institute-of-technology-madras", name=run_name)
    wandb_configs = wandb.config

    #Preparing the data
    img_train = img_train.reshape(img_train.shape[0], -1).T / 255.0
    img_test = img_test.reshape(img_test.shape[0], -1).T / 255.0


    input_neurons = img_train.shape[0]
    output_neurons = len(np.unique(lbl_train))

    one_hot = np.zeros((output_neurons, img_train.shape[1]))

    for i in range(one_hot.shape[1]):

        val = lbl_train[i]

        one_hot[val, i] = 1
    
    # Split the data into training and validation sets
    split_index = int(img_train.shape[1] * 0.9)
    image_val = img_train[:, split_index:]
    image_train = img_train[:, :split_index]
    label_val = lbl_train[split_index:]
    label_train = lbl_train[:split_index]
    one_hot_train = one_hot[:, :split_index]
    one_hot_val = one_hot[:, split_index:]

    # Initialize neural network structure
    fashionnet = NN(input_neurons, output_neurons, wandb_configs)


    m, v = optimization.initialize_m_v(fashionnet) #Initilize the m and v's to be used in optimization algorithms

    num_batches = image_train.shape[1] // wandb_configs.batch_size

    # Train the neural network
    for epoch in range(wandb_configs.epochs):
        # Shuffle the data at the beginning of each epoch
        
        indices = np.random.permutation(image_train.shape[1])
        image_shuffled = image_train[:, indices]
        labels_shuffled = one_hot_train[:, indices]

        #Resets for each epoch
        fashionnet.predictions = []
        fashionnet.activations = {}
        print("Epoch ", end = " ")
        print(epoch)

        t = 1 #Adam timestep set to one for every epoch
       
        for batch in range(num_batches):

            start_index = batch * wandb_configs.batch_size
            end_index = (batch + 1) * wandb_configs.batch_size
            
            # Get batch data and labels
            image_batch = image_shuffled[:, start_index:end_index]
            labels_batch = labels_shuffled[:, start_index:end_index]


            # Forward propagation
            predictions = propagations.forward_propagation(fashionnet, image_batch, wandb_configs)
            

            if wandb_configs.optimizer == "nag":

                for i in range(1, fashionnet.total_layers):
                    index_w = "W" + str(i)
                    index_b = "b" + str(i)

                    fashionnet.parameters[index_w] = fashionnet.parameters[index_w] - wandb_configs.beta * m[index_w]
                    fashionnet.parameters[index_b] = fashionnet.parameters[index_b] - wandb_configs.beta * m[index_b]

            # Backward propagation
            grad_parameters = propagations.backward_propagation(fashionnet, image_batch, labels_batch, predictions, wandb_configs, start_index, end_index, fashionnet.activations["a" + str(fashionnet.total_layers - 1)])
            
            # Update parameters using gradients
            m, v, t = optimization.update_gradients(fashionnet, grad_parameters, wandb_configs, m, v, t)

                
        predictionsv = propagations.forward_propagation(fashionnet, image_val, wandb_configs)
        validation_loss = loss.compute_loss(predictionsv, one_hot_val, wandb_configs.loss, image_val.shape[1], wandb_configs.weight_decay, fashionnet)
                
        predictionsf = propagations.forward_propagation(fashionnet, image_train, wandb_configs)
        training_loss = loss.compute_loss(predictionsf, one_hot_train, wandb_configs.loss, image_shuffled.shape[1], wandb_configs.weight_decay, fashionnet)

        print("Training Loss: ", training_loss)
        print("Validation Loss: ", validation_loss)

        accuracyt = loss.calculate_accuracy(predictionsf, label_train)
        accuracyv = loss.calculate_accuracy(predictionsv, label_val)

        print("Training Accuracy " , accuracyt)
        print("Validation Accuracy ", accuracyv)

        wandb.log({"Training Accuracy": accuracyt, "Validation Accuracy": accuracyv, "Training Loss": training_loss, "Validation Loss": validation_loss, "Epoch": wandb_configs.epochs})
    
    #Test data
    predictionst = propagations.forward_propagation(fashionnet, img_test, wandb_configs)
    test_predictions = np.argmax(predictionst, axis=0)
    class_names = ['T-shirt/top', 
                   'Trouser', 
                   'Pullover', 
                   'Dress', 
                   'Coat', 
                   'Sandal', 
                   'Shirt', 
                   'Sneaker', 
                   'Bag', 
                   'Ankle boot']


    wandb.log({"my_id_conf" : wandb.plot.confusion_matrix(probs=None,
                        y_true=np.reshape(lbl_test,(lbl_test.shape[0])).tolist(), preds=test_predictions.tolist(),
                        class_names=class_names)})

    accuracy_test = loss.calculate_accuracy(predictionst, lbl_test)
    wandb.log({"Test Accuracy" : accuracy_test})
    wandb.run.finish

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-wp', '--wandb_project', default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('-we', '--wandb_entity', default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    parser.add_argument('-d', '--dataset', default="fashion_mnist", help="Dataset choices: [mnist, fashion_mnist]")
    parser.add_argument('-e', '--epochs', default=20, help="Number of epochs to train neural network")
    parser.add_argument('-b', '--batch_size', default=32, help="Batch size used to train neural network")
    parser.add_argument('-l', '--loss', default="cross_entropy", help="Loss function choices: [mean_squared_error, cross_entropy]")
    parser.add_argument('-o', '--optimizer', default="adam", help="Optmizer choices: [sgd, momentum, nag, rmsprop, adam, nadam]")
    parser.add_argument('-lr', '--learning_rate', default=0.001, help="Learning rate used to optimize model parameters")
    parser.add_argument('-m', '--momentum', default=0.8, help="Momentum used by momentum and nag optimizers")
    parser.add_argument('-beta', '--beta', default=0.9, help="Beta used by rmsprop optimizer")
    parser.add_argument('-beta1', '--beta1', default=0.9, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument('-beta2', '--beta2', default=0.999, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument('-eps', '--epsilon', default=1e-8, help="Epsilon used by optimizers")
    parser.add_argument('-w_d', '--weight_decay', default=0.005, help="Weight decay used by optimizers")
    parser.add_argument('-w_i', '--weight_init', default="Xavier", help="Weight initializtion choices: [random, Xavier]")
    parser.add_argument('-nhl', '--num_layers', default=3, help="Number of hidden layers used in feedforward neural network")
    parser.add_argument('-sz', '--hidden_size', default=128, help="Number of hidden neurons in a feedforward layer")
    parser.add_argument('-a', '--activation', default="tanh", help="Activation function choices: [identity, sigmoid, tanh, ReLU]")
    parser.add_argument('-oa', '--output_activation', default="softmax", help="Choice of output activaion function")

    args = parser.parse_args()
    wandb_configs = wbhelp.set_configs(args)
    train(wandb_configs)
