import torch
import numpy
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy
from helper_functions import plot_predictions, plot_decision_boundary

'''
Binary classification with one output variable and n input for the features.
'''
class NeuralNetwork(torch.nn.Module):
    def __init__(self, device, in_features: int,out_features: int, hidden_size=10  ):
        super().__init__()
        print(f"\=== Build Neural Network with {hidden_size} hidden layers\n")
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        ).to(device)

    def forward(self, x):
        return self.layer_stack(x)

'''
A neural network to find the boundaries for random plot following a spiral allocations:
'''

# function to prepare the data using sample moon data from sklearn
def prepare_data(device,n_samples=1000, noise=0.1):
    print("\n=== Prepare a dataset as Tensors fo features and binary class\n")
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples, noise=noise,random_state=42)
    # X[x_coord,y_coord] of a moon in the plane. y=part of red or blue class
    #displayData(X,y)
    X = torch.tensor(X, dtype=torch.float32,device=device)
    y = torch.tensor(y, dtype=torch.float32,device=device)
    return X, y

def displayData(X,y):
    plt.scatter(x=X[:, 0], 
        y=X[:, 1], 
        c=y, 
        cmap=plt.cm.RdYlBu);
    plt.show()


def train(X, y, X_test,y_test, model, loss_fn, optimizer,accuracy_fn):
    epochs=1500
    for epoch in range(epochs):
        model.train()
        # 1. Forward pass on training set - remove last dimension
        y_logits = model(X).squeeze()
        
        loss = loss_fn(y_logits,y)
        y_pred = torch.round(torch.sigmoid(y_logits))
        acc = accuracy_fn(y_pred,y)
        optimizer.zero_grad()
        loss.backward()  # backpropagation
        optimizer.step()

        test_logits, test_pred = test(X_test, model)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(test_pred, y_test)
        if epoch % 100 == 0:
            print(f"\n===== first 5 estimations {y_logits[:5]}\n")
            print(f"\n===== first 5 current classes {y[:5]}\n")
            print(f"Epoch: {epoch} | Loss: {loss:.5f} acc: {acc:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}")
    return model

def test(X_test, model):
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))

    return test_logits,test_pred
                  
def buildTrainTestSets(device,X,y,split):
    print(f"\n=== Build Train and Test Sets at {split*100}% split \n")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split, random_state=42)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    return X_train, X_test, y_train, y_test

def displayBoundary(model,X_train,y_train,X_test,y_test):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)

if __name__ == '__main__':
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    X, y = prepare_data(device)
    print(f"\n=== First 5 Samples of X:\n {X[:5]}")
    print(f"\n=== First 5 Samples of y:\n {y[:5]}")
    
    X_train, X_test, y_train, y_test = buildTrainTestSets(device, X, y, 0.2)

    model= NeuralNetwork(device,2, 2)
    print(model.state_dict())

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)
    # Accuracy from torchmetrics: 
    accuracy_fn = Accuracy(task="binary", num_classes=2).to(device)
    model = train(X_train, y_train, X_test, y_test, model, loss_fn, optimizer, accuracy_fn)
    displayBoundary(model, X_train, y_train, X_test, y_test)
    print("\n=== Do some predictions\n")
    model.eval()
    with torch.inference_mode():
        y_preds = model(X_test).squeeze()
        print(f"Predictions: {y_preds[:10]}\nLabels: {y_test[:10]}")
        print(f"Test accuracy: {accuracy_fn(y_preds, y_test)}%")
    plot_decision_boundary(train_data=X_train, 
                           train_labels=y_train, 
                           test_data=X_test, 
                           test_labels=y_test, 
                           predictions=y_preds)
