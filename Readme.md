# Project Overview

This project contains multiple scripts and modules that work together to define, train, and evaluate a neural network model. Below is a summary of each file and its purpose.

## Files

- **train.py**  
  This script provides the main routine for training the model. It accepts several command-line arguments that allow users to customize the training process.

- **Q1.py**  
  Contains the solution implementation for Question 1.

- **train_sweep.py**  
  Dedicated to setting up and running parameter sweeps on the Weights & Biases (wandb) platform. The overall logic is similar to that in `train.py`.

- **NeuralNetwork.py**  
  Defines the architecture of the neural network and handles the initialization of its parameters.

- **Propagations.py**  
  Implements the logic for both forward and backward propagation within the network.

- **Optimizor4BackProp.py**  
  Contains routines for updating model parameters using various optimization techniques.

- **ActivationdANDLoss.py**  
  - Responsible for computing the loss and its gradient. It also generates run identifiers and constructs a configuration dictionary from the command-line arguments provided to `train.py`.
- Includes functions for applying activation functions to inputs as well as calculating their gradients.

- **CONFIG.py**
  responsible for providing configuration to wandb sweep. It has method for sweep set as basian.

- **confusionMatrix.py**
  contains code for generating confusion matrix for the best model.

- **X_test.npy AND y_test.npy**
  it contains test data and these are used by `confusionMatrix.py` for generating confusion matrix for the best model.

- **best_model.npy**
  it contains model parameter for the best model and used to calculate confusion matrix.

- **fetchSweepID.py**
  As name suggests,it is used to get the sweep id and its corresponding name. it is helpful while creating filter in report.


## How to Run

### 1. Install Required Libraries

Before starting, ensure that the following libraries are installed on your system. Use the command below to install any missing packages:

```bash
pip install numpy keras wandb
```
### 2. Training Process of the Model
- The training process can be started by executing the following command in the terminal:
	- python train.py
- The file train.py takes a number of parameters enabling us to configure the neural network based on our requirements. The below table lists the possible configurations.

|             Name             | Default Value |
| :--------------------------: | :-----------:|
|   -wp, --wandb_project   | myprojectname | 
|   -we, --wandb_entity    |    myname     | 
|      -d, --dataset       | fashion_mnist | 
|       -e, --epochs       |      20       | 
|     -b, --batch_size     |      32       | 
|        -l, --loss        | cross_entropy | 
|     -o, --optimizer      |     adam      | 
|   -lr, --learning_rate   |     0.001     | 
|      -m, --momentum      |      0.5      | 
|      -beta, --beta       |      0.5      | 
|     -beta1, --beta1      |      0.9      | 
|     -beta2, --beta2      |     0.999     | 
|     -eps, --epsilon      |     1e-8      | 
|   -w_d, --weight_decay   |    0.0005     | 
|   -w_i, --weight_init    |    Xavier     | 
|    -nhl, --num_layers    |       3       |
|    -sz, --hidden_size    |      128      | 
|     -a, --activation     |     tanh      | 
| -oa, --output_activation |    softmax    | 

- Suppose we want to execute train.py with 'Sigmoid' activation function we can choose any of the below two commands
	- python train.py --activation sigmoid
- The default values are set to the parameter values which give the best validation accuracy

### 2. Evaluating the model

After training your model, it is essential to verify its performance using proper evaluation techniques. The following steps explain how to evaluate your model both during and after training.


- **Automatic Validation:**  
  During the training process (via `train.py`), the model is periodically evaluated on a validation dataset after each epoch.  
  - **Logged Metrics:** Accuracy, loss, and other performance metrics are automatically logged to wandb
  - **Real-time Monitoring:** Use the wandb dashboard to track the model’s progress and compare various runs and sweeps.



