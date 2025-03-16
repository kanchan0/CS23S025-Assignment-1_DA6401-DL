import numpy as np
import os
import wandb
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, accuracy_score
from ActivationdANDLoss import activations

X_test = np.load("X_test.npy")  
y_test = np.load("y_test.npy")  

if not os.path.exists("best_model.npy"):
    raise RuntimeError("Model file missing......")

model_data = np.load("best_model.npy", allow_pickle=True).item()
weights = model_data["weights"]
activation_func = model_data["activation"]


def activate(Z, func_type):
    if func_type == "ReLU":
        return activations.relu(Z)
    elif func_type == "sigmoid":
        return activations.sigmoid(Z)
    elif func_type == "tanh":
        return activations.tanh(Z)
    else:
        raise ValueError(f"Unknown activation function--> {func_type}")


def make_predictions(X, weights, act_func):
    num_layers = len(weights) // 2
    A = X  

    for layer in range(1, num_layers + 1):
        W, b = weights[f"W{layer}"], weights[f"b{layer}"]
        if W.shape[0] != A.shape[1]:  
            W = W.T  

        Z = A @ W + b.T  
        A = activations.softmax(Z) if layer == num_layers else activate(Z, act_func)

    return np.argmax(A, axis=1)


predictions = make_predictions(X_test, weights, activation_func)
acc = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

class_labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot']


conf_matrix_perc = (conf_matrix / conf_matrix.sum(axis=1, keepdims=True)) * 100
conf_matrix_perc = np.nan_to_num(conf_matrix_perc, nan=0.0)

hover_text = np.empty_like(conf_matrix, dtype=object)
annotations = np.empty_like(conf_matrix, dtype=object)

for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix)):
        if i == j:
            hover_text[i, j] = f"Correct: {conf_matrix[i, j]} ({conf_matrix_perc[i, j]:.1f}%)"
        else:
            hover_text[i, j] = f"Wrong : Was {class_labels[i]} but classified as {class_labels[j]} on {conf_matrix[i, j]} samples"
        
        annotations[i, j] = f"{conf_matrix[i, j]}<br>({conf_matrix_perc[i, j]:.1f}%)"


fig = ff.create_annotated_heatmap(
    z=conf_matrix, 
    x=class_labels, 
    y=class_labels, 
    annotation_text=annotations.tolist(),  
    hovertext=hover_text.tolist(),  
    colorscale="viridis",
    showscale=True
)

fig.update_layout(
    title=f"(Accuracy: {acc:.2%})",
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class",
    autosize=False,  
    width=900,  
    height=900  
)

#fig.show()
wandb.init(project="CS23S025-Assignment-1-DA6401-DL", entity="cs23s025-indian-institute-of-technology-madras")
wandb.log({"Confusion Matrix": wandb.Html(fig.to_html())})  
wandb.finish()
