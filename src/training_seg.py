import mlflow
import torch
import os
import time
import torch.amp
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from src.metrics import Metrics
from src.loss import CombinedLoss

def mlflow_log_hyp(hyperparams: dict):
    """
    Log hyperparameters dict 

    Args:
        hyperparams (dict): Dictionary of hyperparameters (keys) and values to log.

    Returns:
        None
    """
    for key, value in hyperparams.items():
        mlflow.log_param(key, str(value))

def setup_train(model, device, hyperparams: dict):
    
    """
    Configures the model, optimizer, and loss function for segmentation training.

    Args:
        model: PyTorch model instance.
        device: "cuda" or "cpu"
        hyperparams (dict): Dictionary with training parameters.

    Returns:
        tuple: Model, number of epochs, device, loss function, optimizer and scheduler
    """
    assert device.lower() in ["cuda", "cpu"], f"Device must be 'cuda' or 'cpu'"
    model = model.to(device).train()
    epochs = hyperparams["epochs"]
    loss_fn = CombinedLoss(mode="binary")#torch.nn.CrossEntropyLoss()   # Or use DiceLoss + Focal loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparams["lr"])
    scheduler = StepLR(optimizer, step_size=12, gamma=0.1)
    return model, epochs, device, loss_fn, optimizer, scheduler

def move2device(batch, device):
    """
    Moves batches of data (images and labels/mask) to the specified device.

    Args:
        batch (tuple): Batch of data.
        device (str): Device to move the data (e.g., "cuda").

    Returns:
        tuple: Images and labels moved to the device.
    """
    return batch[0].to(device), batch[1].to(device)



def train_one_epoch(model, n_classes, dataloader, device, loss_fn, optimizer, scaler: torch.cuda.amp.GradScaler, epoch):
    """
    Performs training for one epoch with MLflow and segmentation metrics.

    Args:
        model: PyTorch model instance.
        dataloader: DataLoader for training data.
        device (str): Device to use.
        loss_fn: Loss function.
        optimizer: Optimizer instance.
        scaler: GradScaler for FP16 precision.
        epoch (int): Current epoch number.

    Returns:
        dict: Training loss and metrics.
    """

    model.train()
    #Metrric accumulation variables
    train_epoch_loss = 0
    total_miou = 0.0
    total_dice = 0.0
    total_PA = 0.0
    #Instantiate metrics 
    metrics = Metrics(loss_fn, n_classes=n_classes, device=device)
    #Start logging time
    start_time = time.time()

    for batch in tqdm(dataloader, total=len(dataloader)):
        imgs, masks = move2device(batch, device)

        #Mixed precision training
        with torch.amp.autocast(device_type=device):
            preds = model(imgs)            
            loss = loss_fn(preds, masks)

        #Backpropagation and optimizer step, since we are using FP16 for calcs, 
        #we need to ensure a good scale, that's why we use scaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        
        #Compute metrics and accumulate for epoch
        batch_metrics = metrics.compute_metrics(preds, masks)
        total_miou  += batch_metrics["mIoU"]
        total_dice  += batch_metrics["Dice"]
        total_PA    += batch_metrics["Pixel Accuracy"]
        train_epoch_loss += loss.item()

    #Epoch average metrics
    avg_miou  = total_miou / len(dataloader)
    avg_dice  = total_dice / len(dataloader)
    avg_PA    = total_PA / len(dataloader)
    avg_loss  = train_epoch_loss / len(dataloader)

    #Log metrics into MLflow 
    avg_loss = train_epoch_loss / len(dataloader)
    mlflow.log_metric("Train/mIoU", avg_miou, step=epoch)
    mlflow.log_metric("Train/Dice", avg_dice, step=epoch)
    mlflow.log_metric("Train/Pixel Accuracy", avg_PA, step=epoch)
    mlflow.log_metric("Train/Loss", avg_loss, step=epoch)
    mlflow.log_metric("Train/Loss", avg_loss, step=epoch)

    #Log time in MLflow
    end_time = time.time()
    epoch_time = end_time - start_time
    mlflow.log_metric("Train/Time", epoch_time, step=epoch)

    return {"train_loss": avg_loss,
            "train_miou": avg_miou,
            "train_dice": avg_dice,
            "train_PA": avg_PA,            
            }

def val_one_epoch(model, n_classes, dataloader, device, loss_fn, epoch):
    """
    Performs validation for one epoch and tracks metrics.

    Args:
        model: PyTorch model instance.
        dataloader: DataLoader for validation data.
        device (str): Device to use.
        loss_fn: Loss function.
        epoch (int): Current epoch number.

    Returns:
        dict: Validation metrics.
    """
    model.eval()
    
     #Metrric accumulation variables
    total_miou = 0.0
    total_dice = 0.0
    total_PA = 0.0
    val_epoch_loss = 0
    #Instantiate metrics 
    metrics = Metrics(loss_fn, n_classes=n_classes, device=device)
    #Start logging time
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            imgs, masks = move2device(batch,device)
            #Autocast for FP32 or FP16 mixup calcs
            with torch.amp.autocast(device_type=device):
                preds = model(imgs)
                loss = loss_fn(preds, masks)

            batch_metrics = metrics.compute_metrics(preds, masks)
            total_miou  += batch_metrics["mIoU"]
            total_dice  += batch_metrics["Dice"]
            total_PA    += batch_metrics["Pixel Accuracy"]
            val_epoch_loss += loss.item()

    #Mean calculation for metrics
    avg_miou  = total_miou / len(dataloader)
    avg_dice  = total_dice / len(dataloader)
    avg_PA    = total_PA / len(dataloader)
    avg_loss  = val_epoch_loss / len(dataloader)

    #Log metrics into MLflow 
    avg_loss = val_epoch_loss / len(dataloader)
    mlflow.log_metric("Val/mIoU", avg_miou, step=epoch)
    mlflow.log_metric("Val/Dice", avg_dice, step=epoch)
    mlflow.log_metric("Val/Pixel Accuracy", avg_PA, step=epoch)
    mlflow.log_metric("Val/Loss", avg_loss, step=epoch)

    #Logging end time
    end_time = time.time()
    epoch_time = end_time - start_time 
    mlflow.log_metric("Val/Time", epoch_time, step=epoch)

    return {"val_loss": avg_loss,
            "val_miou": avg_miou,
            "val_dice": avg_dice,
            "val_PA": avg_PA
            }
         