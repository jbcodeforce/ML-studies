from fashion_cnn import FashionNISTCNN
import torch, random
from torchvision.transforms import ToTensor 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt



def load_test_fashion_nist(path="data"):
    return datasets.FashionMNIST(
        root=path,
        train=False, # get test data
        download=True,
        transform=ToTensor()
    )

def build_some_samples(data):
    print("\n---- select 9 images")
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(data), k=9):
        test_samples.append(sample)
        test_labels.append(label)
    return test_samples, test_labels

def display_image(class_names,image,label):
    print(image.shape)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(class_names[label])
    plt.axis("Off");
    plt.show()


def make_prediction_on_samples(model: torch.nn.Module, data: list, device: torch.device):
    print(f"\n---- Make Predictions on samples with {device}")
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
    # transform a list into Tensor
    return torch.stack(pred_probs)

def plot_predictions(test_samples, pred_classes, test_labels, class_names):
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
        plt.axis(False)
    plt.show()

def make_prediction_on_test_data(model, data, device):
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X,y in data:
            X= X.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            y_preds.append(y_pred.cpu())
    return torch.cat(y_preds) # transform a list into a single Tensor


def make_confusion_matrix(pred_tensor, test_labels, class_names):
    # Present a confustion matrix between the predicted labels and the true labels from test data
    cm = MulticlassConfusionMatrix(num_classes=len(class_names))
    cm.update(pred_tensor, test_labels)
    fig,ax = cm.plot(labels=class_names)
    plt.show()
    

def getDevice():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")
    return device

if __name__ == "__main__":
    print("\n --- Predictions on 9 image samples\n\n---- Load model")
    device = getDevice()
    model=FashionNISTCNN(input_shape=1,hidden_units=10,output_shape=10)
    model.load_state_dict(torch.load("models/fashion_cnn_model.pth"))
    model.to(device=device)

    test_data = load_test_fashion_nist()
    test_samples, test_labels = build_some_samples(test_data)
    class_names = test_data.classes
    
    #displayImage(class_names, test_samples[0], test_labels[0])
    pred_probs=make_prediction_on_samples(model, test_samples, device)
    print(pred_probs[:2])
    # The predicted label for the sample image is the max of the probability
    pred_labels = pred_probs.argmax(dim=1)
    print(pred_labels,test_labels)
    plot_predictions(test_samples, pred_labels, test_labels,class_names)

    test_data_loader = DataLoader(test_data, batch_size=32)
    y_pred_tensor = make_prediction_on_test_data(model,test_data_loader,device)
    print(y_pred_tensor[:10])
    # pred_tensor includes the predicted labels
    make_confusion_matrix(y_pred_tensor, test_data.targets , class_names)