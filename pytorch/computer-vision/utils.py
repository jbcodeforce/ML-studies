"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random
from typing import List

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  

def display_image(filename:str, cls:str, trans):
    with Image.open(filename) as f:
        if trans != None:
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].imshow(f)
            ax[0].axis(False)
            # permute() changes shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image=trans(f).permute(1,2,0)
            ax[1].set_title(f"New \nSize: {transformed_image.shape}")
            ax[1].imshow(transformed_image)
            ax[1].axis(False)
            fig.suptitle(f"Class: {cls}", fontsize=16) 
        else:
            plt.imshow(f)
            plt.title(f"Class: {cls}",fontsize=16)
            plt.axis(False)
        plt.show()
    
def display_image(img,cls):
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Class: {cls}", fontsize=16)
    plt.axis(False)
    plt.show()

def display_random_images(dataset: torch.utils.data.dataset.Dataset, 
                          classes: List[str],n=4 ):
    
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))
    for i,sample in enumerate(random_samples_idx):
        img, label = data[sample][0], data[sample][1]
        
        plt.subplot(1, n, i+1)
        plt.imshow(img.permute(1, 2, 0))
        plt.axis("off")
        if classes:
            title = f"class: {classes[label]}"
        plt.title(title)
    plt.show()
        