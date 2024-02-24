import torch,os
from pathlib import Path
import torchvision
from torchinfo import summary
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import data_setup, engine,utils
from timeit import default_timer as timer 
from typing import List, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import random


ROOT_FOLDER="./data/pizza_steak_sushi"
IMAGE_SIZE=224
BATCH_SIZE=32
MODEL_FILE_NAME="EfficientNet_B0_pss.pth"

def print_model_summary(model):
    summary(model, input_size=[BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE],
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
            )

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int],
                        transform,
                        device: torch.device):

    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = v2.Compose([
            v2.Resize(image_size),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
    plt.show()


def do_some_predictions(model,transformer,classes,test_dir,device):
    num_images_to_plot = 7
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
    test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        pred_and_plot_image(model,
                            image_path=image_path,
                            class_names=classes,
                            image_size=(IMAGE_SIZE, IMAGE_SIZE),
                            transform=transformer,
                            device=device)
def model_saved():
    return os.path.exists(os.path.join("models",MODEL_FILE_NAME))
      
if __name__ == "__main__":
    test_dir=os.path.join(ROOT_FOLDER, "test")
    train_dir=os.path.join(ROOT_FOLDER, "train")
    device= utils.getDevice()
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    transformer= weights.transforms()
    print("\n--- 1/ load train and test datasets")
    train_dl,test_dl, classes=data_setup.create_data_loaders(
                            train_dir,
                            test_dir,
                            transformer,
                            transformer,
                            batch_size=BATCH_SIZE)
    if model_saved():
        print("Model already saved")
        model=torchvision.models.efficientnet_b0(weights=weights)
        model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True), 
                torch.nn.Linear(in_features=1280, 
                                out_features=len(classes), 
                                bias=True)).to(device)
        model.load_state_dict(torch.load(os.path.join("models", MODEL_FILE_NAME)))
    else:
        print("\n--- 2/ load pre-trained model from PyTorch model")
        model=torchvision.models.efficientnet_b0(weights=weights).to(device)
        print_model_summary(model)
        print("\n--- 3/ train the model")
        for param in model.features.parameters():  # Freeze the features
            param.requires_grad = False
        # Adjust the classifier - output layer
        model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True), 
                torch.nn.Linear(in_features=1280, 
                                out_features=len(classes), 
                                bias=True)).to(device)
        print_model_summary(model)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        start_time = timer()
        results=engine.train(model=model,train_dataloader=train_dl,
                        test_dataloader=test_dl,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=5,
                        device=device)
        end_time = timer()
        print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
        print("\n--- 4/ evaluate the model by plotting loss curves")
        utils.plot_loss_curves(results)
        
        print("\n--- 5/ save the model")
        utils.save_model(model,"models","EfficientNet_B0_pss.pth")
        print("\n--- 6/ make some predictions")
    
    do_some_predictions(model, transformer, classes, test_dir, device)
    print("\n---Done !")
