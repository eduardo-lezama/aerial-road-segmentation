import torch 
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics import Accuracy

class Metrics:
    def __init__(self, loss_fn, n_classes = 2, device= "cpu"):
        """
        Initialize the Metrics class with required parameters.
        Args:
        - loss_fn: Loss function to calculate training loss.
        - n_classes: Number of classes in the segmentation task.
        - device: Device where to do the operations. "cuda" or "cpu"
        """
        self.n_classes = n_classes
        self.loss_fn = loss_fn

        #Initialize torch metrics
        self.miou_metrics = MeanIoU(num_classes=n_classes,).to(device)
        self.dice_metric = DiceScore(num_classes=n_classes).to(device)
        self.pixe_accuracy = Accuracy(task="multiclass", num_classes=n_classes).to(device)
    
    def compute_metrics(self, pred, mask):
        """
        Compute IoU, Dice, and Pixel Accuracy metrics.
        Args:
        - pred: Predicted output from the model (logits or probabilities).
        - gt: Ground truth labels (segmentation mask).
        Returns:
        - Dictionary containing IoU, Dice, and Pixel Accuracy.
        """
        # Convert predictions to class labels if needed
        if pred.ndim == 4:  # Check if predictions have shape (batch, classes, height, width)
            pred = torch.argmax(pred, dim=1)  # Get class with highest probability

        return {
            #.item() because they are tensors
            "mIoU": self.miou_metrics(pred, mask).item(),
            "Dice": self.dice_metric(pred, mask).item(),
            "Pixel Accuracy": self.pixe_accuracy(pred, mask).item()
        }