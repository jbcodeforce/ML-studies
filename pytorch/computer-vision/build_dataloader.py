from torchvision import datasets,transforms
import torch,os,pathlib
from PIL import Image

from typing import Tuple, List, Dict


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    Assumes target directory is in standard image classification format.
    Args:
        directory (str): target directory to load classnames from.
    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")   
    # Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class CustomImageFolder(torch.utils.data.Dataset):

    def __init__(self, target_dir: str, transform=None) -> None:
        self.classes, self.class_to_idx=find_classes(target_dir)
        # Get all image paths
        self.paths = list(pathlib.Path(target_dir).glob("*/*.jpg"))
        self.transform = transform
    
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)
                          
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
    
# Example of how to use it
if __name__ == "__main__":
    root_folder="./data/pizza_steak_sushi"
    trans=transforms.Compose([transforms.Resize((128,128)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
    train_data=CustomImageFolder(os.path.join(root_folder,"train"), transform=trans)
    img, label = train_data[0][0], train_data[0][1]
    print(train_data.class_to_idx)