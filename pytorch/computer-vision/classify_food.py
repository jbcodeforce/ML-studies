import os,argparse,random
import torch,torchvision
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from typing import Dict, List
from TinyVGG import TinyVGG
from data_setup import create_data_loaders
import engine, utils

'''
Build a model to classify food images using 3 classes.
Images are in train or test folder with the specific class
'''
DATA_ROOT_FOLDER="data/"
IMAGE_SIZE=64
def build_folder_name_from_classes(classes):
    return "_".join(classes)  


def trace_dataset(train_data):
    print(train_data)
    print(train_data.class_to_idx)
    img, label = train_data[0][0], train_data[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")
    utils.display_image(img, train_data.classes[label])


def tryOneImage(model,device,train_dl):
    img_batch, label_batch = next(iter(train_dl))
    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    # 3. Perform a forward pass on a single image
    model.eval()
    with torch.inference_mode():
        pred = model(img_single.to(device))
        
    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    plt.show()
  
if __name__ == "__main__":

    
    root_folder=os.path.join(DATA_ROOT_FOLDER, "sushi_steak_pizza")

    print(f"\n---The images for training and test are prepared by prepare_image_dataset.py")
   
    # train_transformer = v2.Compose([v2.Resize((224, 224)),v2.RandomHorizontalFlip(p=0.5),v2.ToTensor()])
    train_transformer=v2.Compose([v2.Resize((IMAGE_SIZE,IMAGE_SIZE)), v2.TrivialAugmentWide(num_magnitude_bins=31), v2.ToTensor()])
    test_transformer=v2.Compose([v2.Resize((IMAGE_SIZE,IMAGE_SIZE)), v2.ToTensor()])
 
    print("\n--- 1: Build dataloader for training and test so neural network can iterate on data")
    train_dl, test_dl, classes = create_data_loaders(os.path.join(root_folder,"train/images"), 
                                          os.path.join(root_folder,"test/images"),
                                          train_transformer,
                                          test_transformer,
                                          batch_size=32)
    print(f"--- The classes to be used for this problem are {classes}")
    device=utils.getDevice()
    NUM_EPOCHS = 10
    print(f"\n--- 2: build and train the model using {NUM_EPOCHS} epochs on {device}") 
    model =TinyVGG(input_shape=3, hidden_units=20, output_shape=len(classes)).to(device)
    try:
        model_info=summary(model, input_size=[1, 3, IMAGE_SIZE, IMAGE_SIZE])
    except UnicodeEncodeError:
        print("problem unicode on windows")
    tryOneImage(model, device,train_dl)
    # Setup loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()
    model_results = engine.train(model=model, 
                        train_dataloader=train_dl,
                        test_dataloader=test_dl,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS,
                        device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    plot_loss_curves(model_results)
    utils.save_model(model, "models", "pizza_steak_sushi_classifier.pth")
