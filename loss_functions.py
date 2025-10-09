import torch

def tube_loss(y_true, y_pred, q=0.95, r=0.1, delta=0.1):
    y_true = y_true.view(-1)
    f1 = y_pred[:, 0]  # upper
    f2 = y_pred[:, 1]  # lower

    c1 = (1 - q) * (y_true - f2)
    c2 = (1 - q) * (f1 - y_true)
    c3 = q * (f2 - y_true)
    c4 = q * (y_true - f1)

    cond1 = (y_true <= f1) & (y_true >= f2)
    inner = torch.where(y_true > r * (f1 + f2), c1, c2)
    outer = torch.where(f2 > y_true, c3, c4)

    loss = torch.where(cond1, inner, outer) + delta * torch.abs(f1 - f2)
    return loss.mean()

def quantile_loss(y_true, y_pred, q=0.5):
    y_true = y_true.view(-1)
    e = y_true - y_pred.squeeze()
    return torch.max(q * e, (q - 1) * e).mean()
