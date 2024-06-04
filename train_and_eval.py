import torch
from torch import nn
from core.utils.distributed import *
import os
import shutil


def train_one_epoch(images, device, imgs_feats, targets, model, criterion, optimizer, lr_scheduler):
    images = images.to(device)
    imgs_feats = imgs_feats.to(device)
    targets = targets.to(device).long()

    outputs = model(images, lbr=imgs_feats)

    loss_dict = criterion(outputs, targets)

    losses = sum(loss for loss in loss_dict.values())

    # reduce losses over all GPUs for logging purposes
    loss_dict_reduced = reduce_loss_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    lr_scheduler.step()

    return losses_reduced


def validation(model, metric, val_loader, device, logger, args):
    order = 0
    best_pred = 0.0
    is_best = False
    metric.reset()
    torch.cuda.empty_cache()
    model.eval()
    for i, (image, target, img_feat) in enumerate(val_loader):
        image = image.to(device)
        img_feat = img_feat.to(device)
        target = target.to(device).long()
        with torch.no_grad():
            outputs = model(image, lbr=img_feat)
        metric.update(outputs[0], target)
    pixAcc, mIoU = metric.get()
    logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
    metric.confusion_plot(id=order)
    order += 1
    new_pred = (pixAcc + mIoU) / 2
    if new_pred > best_pred:
        is_best = True
        best_pred = new_pred
    save_checkpoint(model, args, is_best)
    synchronize()
    print("***********************************")
    # return order,best_pred


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.data_set)
    filename = os.path.join(directory, filename)

    if args.distributed == True:
        torch.save(model.module, filename)
    else:
        torch.save(model, filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.data_set)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
