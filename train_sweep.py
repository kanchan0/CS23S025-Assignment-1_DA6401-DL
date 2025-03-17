import numpy as np
import wandb
from train import wbhelp
from ActivationdANDLoss import loss
from NeuralNetwork import NN
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from Propagations import propagations
from Optimizor4BackProp import optimization
from CONFIG import sweep_configs

hyperparameter_defaults = dict(
    batch_size = 128,
    learning_rate = 0.001,
    epochs = 20,
    n_hidden_layers = 4,
    size_hidden_layers = 64,
    optimizer = "adam"
    )
def train():

    #(img_train, lbl_train), (img_test, lbl_test) = fashion_mnist.load_data()
    (img_train, lbl_train), (img_test, lbl_test) = mnist.load_data()
    #Initialise wandb
    wandb.init(config=hyperparameter_defaults, 
               project="CS23S025-Assignment-1-DA6401-DL",
               entity="cs23s025-indian-institute-of-technology-madras")#, name=sweep_name)
    
    wandb_configs = wandb.config
    sweep_name = wbhelp.sweep_name(wandb_configs)
    wandb.run.name = sweep_name

    #Preparing the data
    img_train = img_train.reshape(img_train.shape[0], -1).T / 255.0
    img_test = img_train.reshape(img_test.shape[0], -1).T / 255.0

    no_of_samples = img_train.shape[1]
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
    m, v = optimization.initialize_m_v(fashionnet)
    num_batches = image_train.shape[1] // wandb_configs.batch_size

    best_validation_accuracy = 0.0
    best_model_weights = None

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
            #print("Batch is " + str(batch))
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

        # Mean loss for the full training set        
        predictionsf = propagations.forward_propagation(fashionnet, image_train, wandb_configs)
        training_loss = loss.compute_loss(predictionsf, one_hot_train, wandb_configs.loss, image_shuffled.shape[1], wandb_configs.weight_decay, fashionnet)

        accuracyt = loss.calculate_accuracy(predictionsf, label_train)
        accuracyv = loss.calculate_accuracy(predictionsv, label_val)

        wandb.log({"Training Accuracy": accuracyt,
                    "Validation Accuracy": accuracyv, 
                    "Training Loss": training_loss,
                    "Validation Loss": validation_loss, 
                    "Epoch": epoch})
         
        # # Save the best model
        # if accuracyv > best_validation_accuracy:
        #     best_validation_accuracy = accuracyv
        #     best_model_weights = fashionnet.get_weights()

        #     # Save model correctly
        #     model_data = {
        #         "weights": best_model_weights,
        #         "activation": wandb.config.activation
        #     }

        #     np.savez("best_model.npz", **model_data)  # ✅ Use np.savez

        #     # Log artifact with correct file name
        #     artifact = wandb.Artifact("best_model_final", type="model")
        #     artifact.add_file("best_model.npz")  # ✅ Correct file name
        #     wandb.log_artifact(artifact)

    wandb.run.finish()

sweep_id = wandb.sweep(sweep_configs, project="CS23S025-Assignment-1-DA6401-DL", entity="cs23s025-indian-institute-of-technology-madras")
wandb.agent(sweep_id, function=train,count=1)