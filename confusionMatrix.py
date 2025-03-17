import numpy as np
import wandb
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, accuracy_score
from ActivationdANDLoss import activations
from Propagations import propagations
from NeuralNetwork import NN
from keras.datasets import fashion_mnist
import plotly.graph_objects as go


wandb.init(project="CS23S025-Assignment-1-DA6401-DL",
           entity="cs23s025-indian-institute-of-technology-madras",
           name="Confusion Matrix Logging")

(_, _), (img_test, lbl_test) = fashion_mnist.load_data()

# flatten and normalize data
img_test = img_test.reshape(img_test.shape[0], -1).T / 255.0
input_size = img_test.shape[0]
output_size = len(np.unique(lbl_test))  

# Load the best model from local storage
best_model_data = np.load("best_model.npz", allow_pickle=True)

# Extract weights and activation function
best_model_weights = best_model_data["weights"].item()
activation_func = best_model_data["activation"]

class DummyWandBConfigs:
    def __init__(self):
        self.hidden_size = 64  
        self.num_layers = 4
        self.weight_init = "Xavier"
        self.activation = activation_func  
        self.output_activation = "softmax"  

dummy_wandb_configs = DummyWandBConfigs()

# Initialize network and load saved weights
fashionnet = NN(input_size, output_size, dummy_wandb_configs)
fashionnet.parameters = best_model_weights  # Load saved weights

# Perform forward propagation
predictions = propagations.forward_propagation(fashionnet, img_test, dummy_wandb_configs)
pred_labels = np.argmax(predictions, axis=0)  # Convert probabilities to class labels

test_accuracy = accuracy_score(lbl_test, pred_labels)
print(f"Final Test Accuracy: {test_accuracy:.2%}")

conf_matrix = confusion_matrix(lbl_test, pred_labels)
conf_matrix_perc = (conf_matrix / conf_matrix.sum(axis=1, keepdims=True)) * 100
conf_matrix_perc = np.nan_to_num(conf_matrix_perc, nan=0.0)

class_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Create Hover Text & Annotations
hover_text = np.empty_like(conf_matrix, dtype=object)
annotations = np.empty_like(conf_matrix, dtype=object)

for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        if i == j:
            hover_text[i, j] = f"✅ Correct: {class_labels[i]}: {conf_matrix[i, j]} ({conf_matrix_perc[i, j]:.1f}%) Correct"
            annotations[i, j] = f"{conf_matrix[i, j]}\n({conf_matrix_perc[i, j]:.1f}%)"
        else:
            hover_text[i, j] = f"❌ Wrong: {class_labels[i]} → {class_labels[j]} ({conf_matrix[i, j]} samples)"
            annotations[i, j] = f"{conf_matrix[i, j]}\n({conf_matrix_perc[i, j]:.1f}%)"
        
        if conf_matrix[i, j] == 0:
            annotations[i, j] = "-"

# Create a 3D Surface Plot for the Confusion Matrix with a new color scheme
fig_3d = go.Figure(data=[go.Surface(z=conf_matrix, 
                                   colorscale= [[0, "pink"], [0.5, "darkgoldenrod"], [1, "green"]])])
fig_3d.update_layout(
    title="3D Confusion Matrix",
    scene=dict(
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        zaxis_title="Count"
    ),
    autosize=False,
    width=950,
    height=900
)
fig_3d.show()
wandb.log({"3D Confusion Matrix": wandb.Html(fig_3d.to_html())})

# Create Confusion Matrix Plot
fig = ff.create_annotated_heatmap(
    z=conf_matrix, 
    x=class_labels, 
    y=class_labels, 
    annotation_text=annotations.tolist(),  
    hovertext=hover_text.tolist(),  
    #colorscale="cividis",
    colorscale = [[0, "lightpink"], [0.5, "darkgoldenrod"], [1, "forestgreen"]],
    showscale=True
)

fig.update_layout(
    title=f"Confusion Matrix (Test Accuracy: {test_accuracy:.1%})",
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class",
    autosize=False,
    width=1000,
    height=900,
    font=dict(size=11)
)

fig.show()

wandb.log({"Confusion Matrix": wandb.Html(fig.to_html())})

# Log Confusion Matrix to WandB as a panel
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    probs=None,
    y_true=lbl_test,
    preds=pred_labels,
    class_names=class_labels
)})

print("Confusion Matrix Logged Successfully!")

wandb.finish()
