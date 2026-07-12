from torch.utils.data import DataLoader
from torchvision import datasets
import os
import torch
from torchvision import transforms


def create_data_loaders(train_dir,test_dir,train_transform,test_transform,batch_size):
    """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    train_transform: torchvision transforms to perform on training 
    test_transform: torchvision transforms to perform on testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             train_transform=some_transform,
                             test_transform=some_transform,
                             batch_size=32
                            )
    """
    print(f"--- Number of CPUs:{os.cpu_count()}")
    train_ds=datasets.ImageFolder(train_dir, transform=train_transform)

    test_ds=datasets.ImageFolder(test_dir,transform=test_transform)
    train_dl=DataLoader(dataset=train_ds, batch_size=batch_size, num_workers=os.cpu_count(),shuffle=True)
    test_dl=DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    return train_dl, test_dl, train_ds.classes

