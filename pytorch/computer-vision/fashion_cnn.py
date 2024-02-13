'''
Loading the nist fashion images, try to do a multi-class classifier using CNN
'''

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor 
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torch import nn
from timeit import default_timer as timer 
import matplotlib.pyplot as plt
import pandas as pd
import random
from pathlib import Path

BATCH_SIZE = 32

def load_training_fashion_nist(path="data"):
    return datasets.FashionMNIST(
        root=path, # where to download data to?
        train=True, # get training data
        download=True, # download data if it doesn't exist on disk
        transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
        target_transform=None # you can transform labels as well
    )

def load_test_fashion_nist(path="data"):
    return datasets.FashionMNIST(
        root=path,
        train=False, # get test data
        download=True,
        transform=ToTensor()
    )


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Basic NN with linear
class FashionNISTLinearModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

# Basic NN with non-linear    
class FashionNISTNonLinearModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units), 
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layer_stack(x)

#  every layer in a neural network is trying to compress data from higher dimensional space to lower dimensional space.
class FashionNISTCNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

def train_step(model: nn.Module, 
               data: DataLoader, 
               loss_fn: nn.Module, 
               optimizer: torch.optim.Optimizer, 
               accuracy_fn: torchmetrics.Accuracy, 
               device: torch.device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_pred, y).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(data)
    train_acc /= len(data)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return train_loss, train_acc

def test_step(model: nn.Module, 
               data: DataLoader, 
               loss_fn: nn.Module, 
               optimizer: torch.optim.Optimizer, 
               accuracy_fn: torchmetrics.Accuracy, 
               device: torch.device):
    test_loss, test_acc = 0, 0 
    model.eval()
    with torch.inference_mode():
        for X, y in test_dl:
            X= X.to(device)
            y=y.to(device)
            # 1. Forward pass
            test_pred = model(X)
           
            # 2. Calculate loss ()
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(test_pred.argmax(dim=1),y)
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dl)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dl)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return test_loss, test_acc


def evaluate_model(model: nn.Module,
                   data: DataLoader,
                   loss_fn: nn.Module,
                   accuracy_fn: torchmetrics.Accuracy,
                   device: torch.device,
                   training_time):
    loss, accuracy = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            accuracy += accuracy_fn(y_pred, y)
        loss /= len(data)
        accuracy /= len(data)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": accuracy.item(),
            "training_time": training_time}

def bestModel(model_0_results, model_1_results, model_2_results):
    compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
    print(compare_results)
    compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
    plt.xlabel("accuracy (%)")
    plt.ylabel("model");
    plt.show()


def train_model(model, train_dl, test_dl):
    torch.manual_seed(42)
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"Train {model.__class__.__name__} using {device} device")
    # create a loss function
    loss_fn = nn.CrossEntropyLoss()

    # create an optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    # Accuracy function
    class_names = train_data.classes
    accuracy_fn=torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)


    # train - test the model
    num_epochs = 3
    train_time_start = timer()
    for epoch in range(num_epochs):
        train_step(model, train_dl, loss_fn, optimizer, accuracy_fn, device)
        test_step(model, test_dl, loss_fn, optimizer, accuracy_fn, device)
    train_time_stop = timer()
    training_time=print_train_time(train_time_start, train_time_stop, device)
    return evaluate_model(model, test_dl, loss_fn, accuracy_fn, device,training_time)

def make_predictions(model, data):
    pred_probs = []
    model.eval()
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) 
            # note: perform softmax on the "logits" dimension, not "batch" dimension 
            # (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
    return torch.stack(pred_probs)

def samplePredictions(model, data):
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(data), k=9):
        test_samples.append(sample)
        test_labels.append(label)
    pred_probs= make_predictions(model=model, 
                             data=test_samples)
    pred_classes = pred_probs.argmax(dim=1)
    plot_predictions(test_samples, pred_classes, test_labels)
    return pred_classes, test_labels

def plot_predictions(test_samples, pred_classes, test_labels):
    class_names = test_samples.classes
    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        # Create a subplot
        plt.subplot(nrows, ncols, i+1)

        # Plot the target image
        plt.imshow(sample.squeeze(), cmap="gray")

        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_names[pred_classes[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_names[test_labels[i]] 

        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        
        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g") # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r") # red text if wrong
        plt.axis(False);

def saveModel(model, filename):
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                    exist_ok=True # if models directory already exists, don't error
    )
    MODEL_SAVE_PATH = MODEL_PATH / filename
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    # load the data
    train_data = load_training_fashion_nist()
    test_data = load_test_fashion_nist()
    train_dl=DataLoader(dataset=train_data,batch_size=BATCH_SIZE, shuffle=True)
    test_dl=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)
    
    # create different models
    model_0 = FashionNISTLinearModel(input_shape=28*28, hidden_units=128, output_shape=10) 
    model_0_results=train_model(model_0, train_dl, test_dl)

    model_1 = FashionNISTNonLinearModel(input_shape=28*28, hidden_units=128, output_shape=10) 
    model_1_results=train_model(model_1, train_dl, test_dl)

    model_2 = FashionNISTCNN(input_shape=1, hidden_units=10, output_shape=10)
    model_2_results=train_model(model_2, train_dl, test_dl)

    bestModel(model_0_results, model_1_results, model_2_results)
   
    samplePredictions(model_2,test_dl)
    saveModel(model_2, "fashion_cnn_model.pth")