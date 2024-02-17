import os,argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''
Build a model to classify food images using 3 classes.
Images are in train or test folder with the specific class
'''
DATA_ROOT_FOLDER="data/"

def build_folder_name_from_classes(classes):
    return "_".join(classes)  
  
def walk_through_folder(folder,class_names):
    # folder: data/pizza_steak_sushi
    train_ds=torch.Tensor()
    test_ds=[]
    train_label_ds=[]
    test_label_ds=[]
    for split in ['test']:
        for cls in class_names:
            dur=os.path.join(folder, split, cls)
            print(f"the folder is {dur}")
            for image in os.listdir(dur):
                print(f"the image is {image}")
                img = np.asarray(Image.open(os.path.join(dur, image)))
                tens=torch.from_numpy(img)
                print(tens.shape)
            #    plt.imshow(img)
            #    plt.show()
            #    break
            #break
            #print(f"the folder is {folder}")
            #print(f"the split is {split}")
            #print(f"the class is {cls}")
            #print(f"the image is {image}")
            #print(f"the image is {os.path.join(folder, split, cls)}")
            #print(f"the image is {os.path.join(folder, split, cls, image)}")
            #print(f"the image is {os.path.join(folder, split, cls, image)}")
    print(train_ds.shape)

def display_image(image:str):
    img = np.asarray(Image.open(image))
    plt.imshow(img)
    plt.show()

def transform_image_to_tensor(image):
    pass

def derive_classes():
    folder=os.listdir(DATA_ROOT_FOLDER)
    return folder[0],folder[0].split('_')

if __name__ == "__main__":
    folder,classes=derive_classes()
    print(f"--- The classes to be used for this problem are {classes}")
    folder=os.path.join(DATA_ROOT_FOLDER, folder)

    print(f"\n---The images for training and test are prepared by prepare_image_dataset.py under {folder}")
    print("--- 1: Prepare data image to tensor for training")
    #walk_through_folder(folder,classes)
    
    #data_folder=os.path.join(DATA_ROOT_FOLDER,build_folder_name_from_classes(classes),"train")
    #print(f"--- 2: Build training data set  {data_folder}")
    display_image(image=folder+"/train/pizza/12301.jpg")
    trans=transforms.Compose([transforms.Resize((128,128)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
    with Image.open(folder+"/train/pizza/12301.jpg") as f:
        img=trans(f)
        print(img.shape)
        plt.imshow(img.permute(1,2,0))
        plt.show()
