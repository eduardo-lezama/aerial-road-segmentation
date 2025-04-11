from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

class CombinedLoss:
    def __init__(self, mode: str, weight_dice: float = 1, weight_focal: float = 0.5):
        """
        Initialize the combined loss function as a weighted sum of DiceLoss and FocalLoss.

        Args:
            mode (str): Mode of operation ('binary', 'multiclass', 'multilabel').
            weight_dice (float): Weight for the Dice loss. Default is 0.5.
            weight_focal (float): Weight for the Focal loss. Default is 0.5.
        """
        self.dice_loss = DiceLoss(mode=mode, from_logits=True)
        self.focal_loss = FocalLoss(mode=mode)
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

    def __call__(self, preds, mask):
        """
        Compute the combined loss.

        Args:
            preds (torch.Tensor): Predictions from the model.
            mask (torch.Tensor): Ground truth segmentation mask.

        Returns:
            torch.Tensor: The weighted sum of DiceLoss and FocalLoss.
        """
        dice = self.dice_loss(preds, mask)
        focal = self.focal_loss(preds, mask)
        return self.weight_dice * dice + self.weight_focal * focal