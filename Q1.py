from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import wandb

(img_train, label_train), (img_test, label_test) = fashion_mnist.load_data()

wandb.init(project="CS23S025-Assignment-1-DA6401-DL", entity="cs23s025-indian-institute-of-technology-madras")

class_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Initialize a list to store one sample image for each class
images = []

# We know that there are 10 different classes in the dataset
for i in range(10):
    for j in range(len(label_train)):
      if i==label_train[j]:
        image = wandb.Image(img_train[j],  caption=f"Class {i} ({class_labels[i]})")
        images.append(image)
        break
      
wandb.log({"Sample_image Q1": images})
wandb.finish()
