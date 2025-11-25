import pytorch_lightning as pl
import torch
from loss.loss import dice_tversky_loss
from metrics.metrics import dice_score, iou_score

class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

    def get_metrics(self):
        # don't show the version number
        items = super().get_metrics()
        items.pop("v_num", None)
        return items

    def _step(self, batch):
        image, y_true = batch
        y_pred, y_pred2, y_pred3, y_pred4 = self.model(image)

        loss1 = dice_tversky_loss(y_pred, y_true)
        loss2 = dice_tversky_loss(y_pred2, y_true)
        loss3 = dice_tversky_loss(y_pred3, y_true)
        loss4 = dice_tversky_loss(y_pred4, y_true)
        loss = loss1*0.4 + loss2*0.3 + loss3*0.2 + loss4*0.1

        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        return loss, dice, iou

    def training_step(self, batch, batch_idx):
        loss, dice_score, iou_score = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice_score, "train_iou": iou_score}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice_score, iou_score = self._step(batch)
        metrics = {"val_loss": loss, "val_dice": dice_score, "val_iou": iou_score}
        self.log_dict(metrics, prog_bar = True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice_score, iou_score = self._step(batch)
        metrics = {"loss":loss, "test_dice": dice_score, "test_iou": iou_score}
        self.log_dict(metrics, prog_bar = True)
        return metrics


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor = 0.5, patience=7, verbose =True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers