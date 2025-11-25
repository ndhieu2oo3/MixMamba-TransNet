import torch

def iou_score(y_pred, y_true):
    smooth = 1e-5
    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()

    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    intersection = (y_pred & y_true).sum()
    union = (y_pred | y_true).sum()

    return (intersection + smooth) / (union + smooth)

def dice_score(y_pred, y_true, smooth=0.):

    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()

    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)