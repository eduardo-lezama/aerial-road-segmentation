import mlflow
import torch
import os
import time
import torch.amp
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from src.metrics import Metrics

def mlflow_log_hyp(hyperparams: dict):
    """
    Log hyperparameters dict 

    Args:
        hyperparams (dict): Dictionary of hyperparameters (keys) and values to log.

    Returns:
        None
    """
    for key, value in hyperparams.items():
        mlflow.log_param(key, value)

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
    loss_fn = torch.nn.CrossEntropyLoss()  # Or use DiceLoss combined with CrossEntropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparams["lr"])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
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
        batch_metrics = Metrics.compute_metrics(preds, masks)
        mlflow.log_metric("Train/IoU", batch_metrics["IoU"], step=epoch)
        mlflow.log_metric("Train/Dice", batch_metrics["Dice"], step=epoch)
        mlflow.log_metric("Train/Pixel Accuracy", batch_metrics["Pixel Accuracy"], step=epoch)

    #Log into MLflow the loss
    avg_loss = epoch_loss / len(dataloader)
    mlflow.log_metric("Train Loss", avg_loss, step=epoch)

    #Log time in MLflow
    end_time = time.time()
    epoch_time = end_time - start_time
    mlflow.log_metric("Train Time", epoch_time, step=epoch)

    return {"train_loss": avg_loss,
            "train_iou": batch_metrics["IoU"],
            "train_dice": batch_metrics["Dice"],
            "train_PA": batch_metrics["Pixel Accuracy"],
            "train_time": epoch_time
            }

def valite_one_epoch(model, dataloader, device, loss_fn, epoch):
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

            val_epoch_loss += loss.items()

            #Cumpute metrics
            batch_metrics = Metrics.compute_metrics(preds, masks)
            mlflow.log_metric("Val/IoU", batch_metrics["IoU"], step=epoch)
            mlflow.log_metric("Val/Dice", batch_metrics["Dice"], step=epoch)
            mlflow.log_metric("Val/Pixel Accuracy", batch_metrics["Pixel Accuracy"], step=epoch)
    
    # Logging end time
    end_time = time.time()
    epoch_time = end_time - start_time 
    mlflow.log_metric("Validation Time", epoch_time, step=epoch)

    #Calculate and log loss
    avg_loss = val_epoch_loss / len(dataloader)
    mlflow.log_metric("Val/Loss", avg_loss, step=epoch)

    return {"val_loss": avg_loss,
            "val_iou": batch_metrics["IoU"],
            "val_dice": batch_metrics["Dice"],
            "val_PA": batch_metrics["Pixel Accuracy"]
            }
         
def save_best_model(model, val_metrics, best_val_loss, experiment_name, run_id, epoch):
    """
    Saves the best model based on validation loss and logs it as an MLflow artifact.

    Args:
        model: PyTorch model instance.
        val_metrics (dict): Validation metrics containing "val_loss".
        best_val_loss (float): Best validation loss so far.
        experiment_name (str): Name of the current experiment.
        run_id (str): MLflow run ID for the current execution.
        epoch (int): Current epoch number.

    Returns:
        float: Updated best_val_loss.
    """
    #Save only if loss is better than previous
    if val_metrics["val_loss"] < best_val_loss:
        os.makedirs("saved_models", exist_ok=True)
        best_val_loss = val_metrics["val_loss"]

        #Save model locally with experiment info in the name
        model_path = f"saved_models/best_model_{experiment_name}_run_{run_id}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)

        #Log the model as a MLflow artifact
        mlflow.log_artifact(model_path)
        print(f"Saved new best model for experiment '{experiment_name}', run {run_id}, epoch {epoch+1}")
# import torch
# import torch.amp
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import precision_score, recall_score
# from torchvision.utils import make_grid
# from torch.amp import GradScaler

# def init_tensorboard_logs(experiment: int, hyperparams: dict):
#     """
#     Initializes TensorBoard and logs hyperparameters.

#     Args:
#         experiment (int): Experiment number.
#         hyperparams (dict): Dictionary of hyperparameters(keys) and values to log.

#     Returns:
#         SummaryWriter: TensorBoard writer object.
#     """
#     #Save the hyperparameters in a text for the current experiment using a writer
#     writer = SummaryWriter(f"logs/experiment{experiment}")
#     formatted_text = "\n".join([f"{key}: {value}" for key, value in hyperparams.items()])
#     writer.add_text("Hyperparameters", formatted_text)
#     return writer

# def setup_train(model, hyperparams: dict):
#     """
#     Configures the model, optimizer, and loss function for training.

#     Args:
#         model: PyTorch model instance.
#         hyperparams (dict): Dictionary with training parameters.

#     Returns:
#         tuple: Model, number of epochs, device, loss function, optimizer.
#     """
#     return model.to("cuda").train(), hyperparams["epochs"], "cuda", torch.nn.CrossEntropyLoss(), torch.optim.Adam(params=model.parameters(), lr=hyperparams["lr"])
#     #Put the model in train() behaviour
#     #7hyperparams["epochs"] --> epochs
#     #"cuda" --> str pointing to cuda device
#     #CrossEntropy as the loss function
#     #Adam as the optimizer with its parameters

# def move2device(batch, device):
#     """
#     Moves batches of data (images and labels) to the specified device.

#     Args:
#         batch (tuple): Batch of data.
#         device (str): Device to move the data (e.g., "cuda").

#     Returns:
#         tuple: Images and labels moved to the device.
#     """
#     return batch[0].to(device), batch[1].to(device)

# def calculate_losses(model, images, labels, loss_fn):
#     """
#     Calculates predictions, loss, and accuracy for a batch.

#     Args:
#         model: PyTorch model instance.
#         images (torch.Tensor): Batch of images.
#         labels (torch.Tensor): Batch of labels.
#         loss_fn: Loss function.

#     Returns:
#         tuple: Loss value, batch accuracy.
#     """
#     preds = model(images)
#     loss = loss_fn(preds, labels)
#     accuracy = (torch.argmax(preds, dim=1) == labels).sum().item()
#     return loss, accuracy

# ### Training and validation loops

# def train_one_epoch(model, dataloader, device, loss_fn, optimizer, scaler: GradScaler, writer: SummaryWriter, epoch):
#     """
#     Performs training for a single epoch.

#     Args:
#         model: PyTorch model instance.
#         dataloader: DataLoader for training data.
#         device (str): Device to use.
#         loss_fn: Loss function.
#         optimizer: Optimizer instance.
#         scaler: GradScaler for FP16 precision.
#         writer: TensorBoard writer.
#         epoch (int): Current epoch number.

#     Returns:
#         dict: Training loss and accuracy metrics.
#     """
#     epoch_loss, epoch_acc = 0, 0
#     model.train() #Put the model into training state

#     for batch in tqdm(dataloader, total=len(dataloader)):
#         imgs, lbls = move2device(batch, device)
#         #Activate autocast for mixed precision
#         with torch.amp.autocast(device):
#             loss, accuracy = calculate_losses(model, imgs, lbls, loss_fn)
#             #Backprop and optimizer step
#             optimizer.zero_grad()
#             #Calculate grads
#             scaler.scale(loss).backward()
#             #Optimizer do an step using those grads
#             scaler.step(optimizer)
#             #Update dynamic scaler
#             scaler.update()
#             #Calculate loss and accuracy per batch and accumulate
#             epoch_loss += loss.item()
#             epoch_acc += accuracy

#     #Register training metrics into TensorBoard
#     #Mean loss per training batch
#     tr_loss_to_track = epoch_loss / len(dataloader)
#     #Mean acc. per epoch
#     tr_acc_to_track = epoch_acc / len(dataloader.dataset)
#     writer.add_scalar("Loss/Train", tr_loss_to_track, epoch)
#     writer.add_scalar("Accuracy/Train", tr_acc_to_track, epoch)

#     #print(f"TRAINING: Epoch: {epoch + 1} train loss: {tr_loss_to_track:.3f}, train accuracy: {tr_acc_to_track:.3f}")
#     return {"train_loss": tr_loss_to_track, "train_accuracy": tr_acc_to_track}

# def validate_one_epoch(model, dataloader, device, loss_fn, writer: SummaryWriter, epoch):
#     """
#     Performs validation for a single epoch and tracks metrics.

#     Args:
#         model: PyTorch model instance.
#         dataloader: DataLoader for validation data.
#         device (str): Device to use.
#         loss_fn: Loss function.
#         writer: TensorBoard writer.
#         epoch (int): Current epoch number.

#     Returns:
#         dict: Validation metrics (loss, accuracy, precision, recall).
#     """
#     model.eval()
#     val_epoch_loss, val_epoch_acc, all_preds, all_labels = 0, 0, [], []

#     with torch.no_grad():
#         for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#             imgs, lbls = move2device(batch, device)
#             #Activate autocast for validation (FP16)
#             with torch.amp.autocast(device):
#                 preds = model(imgs)
#                 loss = loss_fn(preds, lbls)

#             val_epoch_loss += loss.item()
#             #From preds (for each image has a tensor of 10 logits), get the total of corrects predictions
#             val_epoch_acc += (torch.argmax(preds, dim=1) == lbls).sum().item()
#             #Accumulation of preds and labels, to calculate at the end of the val.
#             #Calculating precisio and recall per batch could cause problems as there could be batches without
#             #certain type of images.
#             all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
#             all_labels.extend(lbls.cpu().numpy())

#             #Save images from the first batch only, as samples
#             if i == 0:
#                 grid = make_grid(imgs[0:8].cpu(), nrow=2, normalize=True)
#                 writer.add_image(f"Validation examples/Epoch {epoch+1}", grid, epoch)
#                 writer.add_text(f"Predictions/Epoch {epoch+1}",
#                                 f"Predicted: {torch.argmax(preds[:8], dim=1).cpu().numpy()}, "
#                                 f"Actual: {lbls[:8].cpu().numpy(), epoch}", 
#                                 epoch + 1
#                                 )

#     #Register training metrics into TensorBoard
#     #Mean loss per batch
#     val_loss_to_track = val_epoch_loss / len(dataloader) # len(val_dl) is the total number of batches
#     #Mean accuracy of the val dataset
#     val_acc_to_track = val_epoch_acc / len(dataloader.dataset) #The reason we use dataset is because it is calculated per image
#     #Mean precision and recall using all preds and labels from the validation cycle.
#     precision  = precision_score(all_preds, all_labels, average="weighted")
#     recall  = recall_score(all_preds, all_labels, average="weighted")
#     #Write and register the metrics
#     writer.add_scalar("Loss/Validation", val_loss_to_track, epoch)
#     writer.add_scalar("Accuracy/Validation", val_acc_to_track, epoch)
#     writer.add_scalar("Precision/Validation", precision, epoch)
#     writer.add_scalar("Recall/Validation", recall , epoch)
    
#     return {"val_loss": val_loss_to_track, 
#             "val_accuracy": val_acc_to_track, 
#             "precision": precision, 
#             "recall": recall}