sweep_configs = {

            "name": "DL-Assignment-1_Sweep-8",
            "metric": {
                        "name":"Validation Accuracy",
                        "goal": "maximize"
                      },
            "method": "bayes",
            "early_terminate": {
                        "type": "hyperband",
                        "min_iter": 2,
                        "eta" : 2
                      },
            "parameters": {
                    "hidden_size": {
                        "values": [32, 64, 128]
                    },
                    "num_layers": {
                        "values": [3,4, 5]
                    },
                    "activation": {
                        "values": ["sigmoid", "tanh", "ReLU"]
                    },
                    "optimizer": {
                        "values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
                    },
                    "epochs": {
                        "values": [5,15 ,10, 20]
                    },
                    "batch_size": {
                        "values": [8, 16, 64, 128]
                    },
                    "learning_rate": {
                        "values": [0.001, 0.0001,0.01]
                    },
                    "weight_init": {
                        "values": ["random", "Xavier"]
                    },
                    "weight_decay": {
                        "values": [0.005, 0.05]
                    },
                    "loss": {
                        "values": ["cross_entropy"]
                        #"values" :["mean_squared_error"]
                    },
                    "output_activation": {
                        "values": ["softmax"]
                    },
                    "momentum": {
                        "values": [0.9]
                    },
                    "beta": {
                        "values": [0.9]
                    },
                    "beta1": {
                        "values": [0.9]
                    },
                    "beta2": {
                        "values": [0.999]
                    },
                    "epsilon": {
                        "values": [1e-8]
                    }
            }
            }
