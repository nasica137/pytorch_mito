import torch
from pprint import pprint
import pandas as pd
#import monai
from monai.data import Dataset, DataLoader
from monai.transforms import ToTensor, Compose
from monai.networks.nets import UNet
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
import monai.losses as monai_losses
from monai.losses import DiceLoss
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
os.environ['TORCH_HOME'] = '/misc/lmbraid21/nasica/tmp'
from my_utils import TrainEpoch, ValidEpoch, NamedLoss
from custom_transform import CustomTransform, ConvertToMultiChannelMask


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import json
import segmentation_models_pytorch as smp
from monai.losses import DiceLoss
from torch.optim import Adam
import argparse

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Perform finetuning on a large image.')
parser.add_argument('--config', type=str, help='Path to the configuration JSON file.')
args = parser.parse_args()



# Load configuration from JSON file
with open(args.config, 'r') as config_file:
    config = json.load(config_file)
    



# Access configuration parameters
data_dir = config["data_dir"]
images_dir = os.path.join(data_dir, config["images_dir"])
masks_dir = os.path.join(data_dir, config["masks_dir"])
#val_images_dir = os.path.join(data_dir, "val/images")
#val_masks_dir = os.path.join(data_dir, "val/masks")
train_size = config["train_size"]
val_size = config["val_size"]
test_size = config["test_size"]
batch_size_train = config["batch_size"]["train"]
batch_size_val = config["batch_size"]["val"]
batch_size_test = config["batch_size"]["test"]
num_workers_train = config["num_workers"]["train"]
num_workers_val = config["num_workers"]["val"]
num_workers_test = config["num_workers"]["test"]
spatial_size = config["spatial_size"]
pos = config["pos"]
neg = config["neg"]
num_samples = config["num_samples"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]

# Model architecture configuration
model_config = config["model"]
model_name = model_config["name"]
encoder_name = model_config["encoder_name"]
encoder_weights = model_config["encoder_weights"]
in_channels = model_config["in_channels"]
classes = model_config["classes"]

# Loss function configuration
loss_config = config["loss"]
loss_name = loss_config["name"]

# Optimizer configuration
optimizer_config = config["optimizer"]
optimizer_name = optimizer_config["name"]


# Define the output directory based on the configuration
output_directory = config["output_directory"].format(**config)

# Create the directory if it doesn't exist
os.makedirs('fine-tuning/' + output_directory, exist_ok=True)



from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')

# Set a new temporary directory
new_tmpdir = '/misc/lmbraid21/nasica/tmp'  # Replace this with your desired temporary directory

# Set the environment variable
os.environ['TMPDIR'] = ''
os.environ['TEMP'] = ''

from multiprocessing import Manager
mp = Manager()
mp.shutdown()


#-----------------------------------------------------------------------------------------------------------------------
# Dataloading and Preprocessing
#-----------------------------------------------------------------------------------------------------------------------

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Rand2DElasticd,
    RandAdjustContrastd,
    AsDiscreted,
    RandZoomd,
    RandAffined,
    Lambdad,
    Resized
)

#data_dir = "../mitochondria/data2"
#images_dir = os.path.join(data_dir, "images")
#masks_dir = os.path.join(data_dir, "masks_png")

images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(".png")]
masks = [os.path.join(masks_dir, mask) for mask in os.listdir(masks_dir) if mask.endswith(".png")]

images.sort()
masks.sort()


train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),  
    RandCropByPosNegLabeld(
        keys=["image", "mask"],
        label_key="mask",
        spatial_size=spatial_size,
        pos=pos,
        neg=neg,
        num_samples=num_samples,
        image_key="image",
    ),
    CustomTransform(keys=["image"], preprocess_input=preprocess_input),
    RandFlipd(keys=["image", "mask"], prob=0.5),
    RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=[0, 1]),
    RandZoomd(keys=["image", "mask"], prob=1.0, mode="nearest", padding_mode='constant', min_zoom=0.7, max_zoom=1.2),
    Rand2DElasticd(keys=["image", "mask"], prob=0.7, spacing=20, magnitude_range=(1, 2)),
    RandGaussianNoised(keys=["image"], prob=0.6, mean=0.0, std=0.1),
    RandGaussianSmoothd(keys=["image"], prob=0.3),  
    RandAdjustContrastd(keys=["image"], prob=1.0, gamma=(0.1, 5.5)),
    AsDiscreted(threshold=0.001, keys=["mask"]),
])


val_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),   
    EnsureChannelFirstd(keys=["image", "mask"]),  
    RandCropByPosNegLabeld(
        keys=["image", "mask"],
        label_key="mask",
        spatial_size=spatial_size,
        pos=pos,
        neg=neg,
        num_samples=num_samples,
        image_key="image",
    ),
    CustomTransform(keys=["image"], preprocess_input=preprocess_input),
    AsDiscreted(threshold=0.001, keys=["mask"]),
    #ConvertToMultiChannelMask(keys=["mask"]),  # Apply the custom transform
])


# Define your validation and test files
train_files = [{"image": img, "mask": mask} for img, mask in zip(images[:4], masks[:4])]
val_files = [{"image": img, "mask": mask} for img, mask in zip(images[4:5], masks[4:5])]


# Use the same transformations as the training set for both validation and test datasets
train_ds = Dataset(data=train_files, transform=train_transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)


# Create validation and test data loaders
train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=num_workers_train)
val_loader = DataLoader(val_ds, batch_size=batch_size_val, num_workers=num_workers_val)


# Report split sizes
print('Training set has {} instances'.format(len(train_ds)))
print('Validation set has {} instances'.format(len(val_ds)))

#-----------------------------------------------------------------------------------------------------------------------
# Sanity Check
#-----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from random import randint

# Assuming you have a DataLoader named train_loader
batch = next(iter(train_loader))  # Fetching a batch

# Show and save all the accumulated plots
fig, axes = plt.subplots(6, 2, figsize=(10, 20))  # Adjust figsize as needed

# Set the average IoU scores and AP values as the title
plt.suptitle(f"Sanity Check", fontsize=16, y=0.95)

for i in range(6):  # Plotting six samples
    images = batch['image']
    masks = batch['mask']
    images = torch.mean(images, dim=1)
    print(images.shape)
    # Add the generated plots to the corresponding subplots
    axes[i, 0].imshow(images[i].squeeze(), cmap='gray')
    axes[i, 0].set_title("Image")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
    axes[i, 1].set_title("Ground Truth")
    axes[i, 1].axis('off')

plt.savefig(f"./fine-tuning/{output_directory}/six_samples.png", bbox_inches='tight', pad_inches=0.1)


#-----------------------------------------------------------------------------------------------------------------------
# Testing
#-----------------------------------------------------------------------------------------------------------------------

# Load the pre-trained model
pretrained_model_path = f'./pretrained/{output_directory}/best_model.pth'
model = torch.load(pretrained_model_path)

# Unwrap the model if it was wrapped with DataParallel
if isinstance(model, torch.nn.DataParallel):
    model = model.module

# Adapt the last layer for multiclass segmentation (assuming 3 classes)
model.segmentation_head[0] = torch.nn.Conv2d(
    in_channels=model.segmentation_head[0].in_channels,
    out_channels=1,  # Change this to the number of classes in your multiclass task
    kernel_size=1
)

# Move the model to the available devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move the model to the first GPU if available
model = torch.nn.DataParallel(model)  # Wrap with DataParallel


# Loss function definition based on configuration
loss_fn = getattr(monai_losses, loss_name)(sigmoid=True)

# Wrap the loss function with a name
loss_fn = NamedLoss(loss_fn, loss_name)

# define metrics
metrics = [
    smp_utils.metrics.IoU(threshold=0.5),
    smp_utils.metrics.Accuracy(threshold=0.5),
]

# Optimizer definition based on configuration
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)



train_epoch = TrainEpoch(
    model, 
    loss=loss_fn, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
    verbose=True,
)

valid_epoch = ValidEpoch(
    model, 
    loss=loss_fn, 
    metrics=metrics, 
    device=device,
    verbose=True,
)
     
best_iou_score = 0.0
train_logs_list, valid_logs_list = [], []

for i in range(0, epochs):

    # Perform training & validation
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(val_loader)
    train_logs_list.append(train_logs)
    valid_logs_list.append(valid_logs)

    # Save model if a better val IoU score is obtained
    if best_iou_score < valid_logs['iou_score']:
        best_iou_score = valid_logs['iou_score']
        torch.save(model, f'./fine-tuning/{output_directory}/best_model.pth')
        print('Model saved!')
    

print(f"Finished finetuning {encoder_name} with {loss_name}")
#-----------------------------------------------------------------------------------------------------------------------
# Plotting
#-----------------------------------------------------------------------------------------------------------------------

train_logs_df = pd.DataFrame(train_logs_list)
valid_logs_df = pd.DataFrame(valid_logs_list)

train_logs_df.to_csv(f'./fine-tuning/{output_directory}/train_logs.csv', index=False)
valid_logs_df.to_csv(f'./fine-tuning/{output_directory}/valid_logs.csv', index=False)


# metrics
plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('IoU Score', fontsize=20)
plt.title('IoU Score Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(f'./fine-tuning/{output_directory}/iou_score_plot.png')

# loss
plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df[loss_name].tolist(), lw=3, label = 'Train')
plt.plot(valid_logs_df.index.tolist(), valid_logs_df[loss_name].tolist(), lw=3, label = 'Valid')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title(f'{loss_name} Plot', fontsize=20)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig(f'./fine-tuning/{output_directory}/{loss_name}_plot.png')
