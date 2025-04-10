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
    loss_fn = CombinedLoss(mode="binary") #torch.nn.CrossEntropyLoss()  # Or use DiceLoss + Focal loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparams["lr"])
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
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



def train_one_epoch(model, dataloader, device, loss_fn, optimizer, scaler: torch.cuda.amp.GradScaler, epoch):
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
    epoch_loss = 0
    model.train()

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

        epoch_loss += loss.item()

        #Compute metrics and log into MLflow
        metrics = Metrics(loss_fn, device=device)
        batch_metrics = metrics.compute_metrics(preds, masks)
        mlflow.log_metric("Train/IoU", batch_metrics["mIoU"], step=epoch)
        mlflow.log_metric("Train/Dice", batch_metrics["Dice"], step=epoch)
        mlflow.log_metric("Train/Pixel Accuracy", batch_metrics["Pixel Accuracy"], step=epoch)

    #Log into MLflow the loss
    avg_loss = epoch_loss / len(dataloader)
    mlflow.log_metric("Train/Loss", avg_loss, step=epoch)

    #Log time in MLflow
    end_time = time.time()
    epoch_time = end_time - start_time
    mlflow.log_metric("Train/Time", epoch_time, step=epoch)

    return {"train_loss": avg_loss,
            "train_miou": batch_metrics["mIoU"],
            "train_dice": batch_metrics["Dice"],
            "train_PA": batch_metrics["Pixel Accuracy"],
            "train_time": epoch_time
            }

def val_one_epoch(model, dataloader, device, loss_fn, epoch):
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
    val_epoch_loss = 0

    #Start logging time
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            imgs, masks = move2device(batch,device)
            #Autocast for FP32 or FP16 mixup calcs
            with torch.amp.autocast(device_type=device):
                preds = model(imgs)
                loss = loss_fn(preds, masks)

            val_epoch_loss += loss.item()

            #Cumpute metrics
            metrics = Metrics(loss_fn, device=device)
            batch_metrics = metrics.compute_metrics(preds, masks)
            mlflow.log_metric("Val/IoU", batch_metrics["mIoU"], step=epoch)
            mlflow.log_metric("Val/Dice", batch_metrics["Dice"], step=epoch)
            mlflow.log_metric("Val/Pixel Accuracy", batch_metrics["Pixel Accuracy"], step=epoch)
    
    # Logging end time
    end_time = time.time()
    epoch_time = end_time - start_time 
    mlflow.log_metric("Val/Time", epoch_time, step=epoch)

    #Calculate and log loss
    avg_loss = val_epoch_loss / len(dataloader)
    mlflow.log_metric("Val/Loss", avg_loss, step=epoch)

    return {"val_loss": avg_loss,
            "val_miou": batch_metrics["mIoU"],
            "val_dice": batch_metrics["Dice"],
            "val_PA": batch_metrics["Pixel Accuracy"]
            }
         
# def save_best_model(model, val_metrics, best_val_loss, experiment_name, run_id, epoch):
#     """
#     Saves the best model based on validation loss and logs it as an MLflow artifact.

#     Args:
#         model: PyTorch model instance.
#         val_metrics (dict): Validation metrics containing "val_loss".
#         best_val_loss (float): Best validation loss so far.
#         experiment_name (str): Name of the current experiment.
#         run_id (str): MLflow run ID for the current execution.
#         epoch (int): Current epoch number.

#     Returns:
#         float: Updated best_val_loss.
#     """
#     #Save only if loss is better than previous
#     if val_metrics["val_loss"] < best_val_loss:
#         os.makedirs("saved_models", exist_ok=True)
#         best_val_loss = val_metrics["val_loss"]

#         #Save model locally with experiment info in the name
#         model_path = f"saved_models/best_model_{experiment_name}_run_{run_id}.pth"
#         torch.save(model.state_dict(), model_path)

#         #Log the model as a MLflow artifact
#         mlflow.log_artifact(model_path)
#         print(f"Saved new best model for experiment '{experiment_name}', run {run_id}, epoch {epoch+1}")