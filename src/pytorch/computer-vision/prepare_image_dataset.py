'''
Prepare a subset of images from a big dataset.
The data set is food 101 from PyTorch vision https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/ 
with only the selected classes: sushi,steak,pizza
'''
import argparse
import os
import shutil
import random
import torchvision.datasets as ds

DATA_ROOT_FOLDER="data/"
DATA_FOOD_FOLDER="food-101/"
SAMPLE_AMOUNT=0.2



def process_program_arguments():
    parser = argparse.ArgumentParser(description='Prepare a subset of images from a big dataset')
    parser.add_argument('--classes', type=str, help='comma separated list of classes')
    args = parser.parse_args()
    classes = args.classes.split(',')
    return classes


def build_target_structure(data_folder,classes):
    '''
    under DATA folder build a folder with a name = class names concatenated, and train and test folders 
    under it.
    '''
    os.makedirs(data_folder, exist_ok=True)
    dst=os.path.join(data_folder,"_".join(classes))
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(dst, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst, "test"), exist_ok=True)
    return dst
    
def download_food_101_dataset(data_folder,download: False):
    """
    Download the 4GB of photos to data_folder and then create a folder
    data/sushi_meat_pizza for example with train and test subfolders
    """
    os.makedirs(data_folder, exist_ok=True)
    ds.Food101(root=data_folder, split='train', download=download)
    ds.Food101(root=data_folder, split='test', download=download)

def prepare_data_subset(food_src,classes):
    '''
    Copy images from the original dataset to the new one for each class
    dst/class_name/train
    dst/class_name/test
    images for a given classes are in food_src/food_src/images/class_name
    The food_src/food_src/meta folder has two files to present the list of file for training and test sets
    '''
    dataset_split = ['train', 'test']
    label_splits = {} # the list of labels per split
    for split in dataset_split:
        label_path = os.path.join(food_src, food_src, 'meta', split + '.txt')
        with open(label_path, 'r') as f:
            labels = [line.strip("\n") for line in f.readlines() if line.split("/")[0] in classes]
            print(labels)
        nb_of_samples= round(SAMPLE_AMOUNT*len(labels))
        sampled_images=random.sample(labels, k=nb_of_samples)
        image_paths=[]
        for sample_image in sampled_images:
            image_path = os.path.join(food_src, food_src, 'images', sample_image)+ ".jpg"
            print(image_path)
            image_paths.append(image_path)
        label_splits[split]=image_paths
    return label_splits


def move_data_clean(src,label_splits,dst):
    for split in label_splits:
        for image_path in label_splits[split]:
            aclass=image_path.split("/")[-2]
            aimage=image_path.split("/")[-1]
            dst2=os.path.join(dst, split, aclass, aimage)
            os.makedirs(os.path.join(dst, split, aclass), exist_ok=True)
            shutil.copy2(image_path, dst2)
    os.remove(src)


if __name__ == '__main__':
    classes = process_program_arguments()
    dst=build_target_structure(DATA_ROOT_FOLDER,classes)
    download_food_101_dataset(DATA_FOOD_FOLDER,True)
    label_splits=prepare_data_subset(DATA_FOOD_FOLDER,classes)
    print(label_splits["train"][:10])
    print(label_splits["test"][:5])
    move_data_clean(DATA_FOOD_FOLDER,label_splits,dst)