from fashion_cnn import FashionNISTCNN
import torch
from torchvision.transforms import ToTensor 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def load_test_fashion_nist(path="data"):
    return datasets.FashionMNIST(
        root=path,
        train=False, # get test data
        download=True,
        transform=ToTensor()
    )

if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    model=FashionNISTCNN(input_shape=1,hidden_units=10,output_shape=10)
    model.load_state_dict(torch.load("models/fashion_cnn_model.pth"))
    model.to(device=device)
    test_data = load_test_fashion_nist()
