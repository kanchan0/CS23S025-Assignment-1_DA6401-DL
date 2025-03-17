sweep_configs = {

            "name": "DL-Assignment-1_sweepforCOnfusionMatrix",
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
                        "values": [64]
                    },
                    "num_layers": {
                        "values": [4]
                    },
                    "activation": {
                        "values": ["ReLU"]
                    },
                    "optimizer": {
                        "values": ["adam"]
                    },
                    "epochs": {
                        "values": [20]
                    },
                    "batch_size": {
                        "values": [128]
                    },
                    "learning_rate": {
                        "values": [0.001]
                    },
                    "weight_init": {
                        "values": ["Xavier"]
                    },
                    "weight_decay": {
                        "values": [0.005]
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
