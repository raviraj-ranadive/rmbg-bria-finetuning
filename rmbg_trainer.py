import os
import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from skimage import io
from skimage.color import rgba2rgb, gray2rgb
from torch.utils.data import DataLoader
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from example_inference import model_inference
from torch.utils.data import DataLoader, random_split



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, device):
        self.image_paths_list = image_dir
        self.mask_paths_list = mask_dir
        self.device = device
        self.model_input_size = [1024, 1024]

    def preprocess_image(self, im: np.ndarray, model_input_size: list) -> torch.Tensor:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)  # CxHxW
        im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=model_input_size, mode='bilinear', align_corners=False)
        im_tensor = im_tensor.squeeze(0).type(torch.float32)
        image = im_tensor / 255.0
        image = transforms.Normalize([0.485, 0.456, 0.456], [0.229, 0.224, 0.225])(image)
        return image

    def preprocess_mask(self, mask: np.ndarray, model_input_size: list) -> torch.Tensor:
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, :, :]  # Adding channel dimension
        mask_tensor = torch.tensor(mask, dtype=torch.float32)   # CxHxW
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=model_input_size, mode='nearest').squeeze(0)  # remove align_corners
        mask_tensor = (mask_tensor > 0.5).float()  # binarization
        return mask_tensor

    def __len__(self):
        return len(self.image_paths_list)

    def __getitem__(self, idx):
        img_path = self.image_paths_list[idx]
        mask_path = self.mask_paths_list[idx]

        orig_im = io.imread(img_path)
        # Convert to RGB if image is grayscale
        if len(orig_im.shape) == 2 or orig_im.shape[2] == 1:
            orig_im = gray2rgb(orig_im)

        # Convert to RGB if image is RGBA
        if orig_im.shape[2] == 4:
            orig_im = rgba2rgb(orig_im)

        orig_mask = io.imread(mask_path, as_gray=True)

        image = self.preprocess_image(orig_im, self.model_input_size).to(self.device)
        mask = self.preprocess_mask(orig_mask, self.model_input_size).to(self.device)

        return image, mask
    



def log_evaluation_images(model, image_paths, true_mask_paths, epoch):
    model.eval()
    with torch.no_grad():
        image_np_list = []
        mask_np_list = []
        pred_mask_np_list = []
        for i, (img_path, mask_path) in enumerate(zip(image_paths, true_mask_paths)):
            # Load image and mask
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            img_np = np.array(img)
            mask_np = np.array(mask)
            # Get predicted mask from example_inference
            pred_mask_np, removed_bkg = model_inference(img_path, model)

            image_np_list.append(img_np)
            mask_np_list.append(mask_np)
            pred_mask_np_list.append(pred_mask_np)
        model.train()
        return image_np_list, mask_np_list, pred_mask_np_list 


def finetune_model(image_dir, mask_dir, model_path, epochs, batch_size, learning_rate):
    wandb.init(project="Experiment-4-rmbg")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and dataloader
    dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir, device=device)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load local weights (.pth)
    model = BriaRMBG()
    if model_path:
        print("Loading pretrained model from:", model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-07, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.BCEWithLogitsLoss()
    
    # Lists to store losses for wandb logging
    train_losses = []
    val_losses = []
    
    # Train loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Wrap DataLoader with tqdm for progress bar
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as train_loader_desc:
            for images, masks in train_loader_desc:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)[0][0]
                
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Update tqdm description with current loss
                train_loader_desc.set_postfix(loss=loss.item())
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        # Log the training loss to wandb
        wandb.log({"epoch": epoch+1,})
        
        print(f'Epoch {epoch+1}/{epochs}, Avg Train Loss: {avg_epoch_loss}')
        
        # Evaluation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)[0][0]
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Log the validation loss to wandb
        wandb.log({"epoch": epoch+1, })
        
        print(f'Epoch {epoch+1}/{epochs}, Avg Val Loss: {avg_val_loss}')
        image_np_list, mask_np_list, pred_mask_np_list = log_evaluation_images(model, eval_img_list, eval_masks_list, epoch)
        wandb.log({
            "epoch": epoch+1,
            "Loss": {"train_loss":avg_epoch_loss, "validation_loss" : avg_val_loss},
            "Original Image": [wandb.Image(img) for img in image_np_list],
            "Bria Mask": [wandb.Image(bria_mask) for bria_mask in mask_np_list],
            "Predicted Mask": [wandb.Image(pred_mask) for pred_mask in pred_mask_np_list]
        })
        
        torch.save(model.state_dict(), f'./weight_files_exp_4/finetuned_model-ep{epoch}.pth')
        
        lr_scheduler.step()
        
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        wandb.log({"epoch": epoch+1, "learning_rate": current_lr})
    
    torch.save(model.state_dict(), './weight_files_exp_4/final-finetuned-model.pth')
    del model


def main():
    # Load datasets
    df_cleaned_train = pd.read_csv("./dataset-csv/complete_dataset.csv")

    # Shuffle the DataFrame rows
    df_cleaned_train = df_cleaned_train.sample(frac=1, random_state=48).reset_index(drop=True)

    evaluation_df = pd.read_csv("./dataset-csv/eval_data.csv")

    images_paths_list = df_cleaned_train["image_path"]
    masks_paths_list = df_cleaned_train["mask_path"]

    global eval_img_list, eval_masks_list
    eval_img_list = evaluation_df["images_path"]
    eval_masks_list = evaluation_df["masks_path"]

    # Launch FineTuning
    finetune_model(image_dir=images_paths_list,
                   mask_dir=masks_paths_list,
                   model_path="/media/pixis/pixis/Ravi_workspace/rembg-finetune/model.pth",
                   epochs=150,
                   batch_size=8,
                   learning_rate=1e-3)

if __name__ == "__main__":
    main()
