import torch
from config.segmentor import Segmentor
import pytorch_lightning as pl
import os
from module.model.mixmambatransnet import MixMamba_TransNet
from datasets.datasets import trainloader, valloader
from module.modules import pvt_v2_b5

import csv
class HistoryLogger(pl.callbacks.Callback):
    def __init__(self, dir = "history_rivf.csv"):
        self.dir = dir
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "loss_epoch" in metrics.keys():
            logs = {"epoch": trainer.current_epoch}
            keys = ["loss_epoch", "train_dice_epoch", "val_loss","val_dice"
                    ]
            for key in keys:
                logs[key] = metrics[key].item()
            header = list(logs.keys())
            isFile = os.path.isfile(self.dir)
            with open(self.dir, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if not isFile:
                    writer.writeheader()
                writer.writerow(logs)
        else:
            pass

backbone = pvt_v2_b5()
backbone.load_state_dict(torch.load(PRETRAINED_PATH)) ## pvt_v2_b5 pretrained weights

model = MixMamba_TransNet(backbone=backbone)
model = MixMamba_TransNet().cuda()

os.makedirs('/content/drive/MyDrive/weight_polyp', exist_ok = True)
check_point = pl.callbacks.model_checkpoint.ModelCheckpoint("/content/drive/MyDrive/weight_polyp", filename="ckpt{val_dice:0.4f}-v2.2",
                                                            monitor="val_dice", mode = "max", save_top_k =1,
                                                            verbose=True, save_weights_only=True,
                                                            auto_insert_metric_name=False,)
progress_bar = pl.callbacks.TQDMProgressBar()
logger = HistoryLogger()
PARAMS = {"benchmark": True, "enable_progress_bar" : True,"logger":False,
        #   "callbacks" : [progress_bar],
        #    "overfit_batches" :1,
          "callbacks" : [check_point, progress_bar],
          "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":250,
          "precision":16,
          }


trainer = pl.Trainer(**PARAMS)

segmentor = Segmentor(model=model)
trainer.fit(segmentor, trainloader, valloader)