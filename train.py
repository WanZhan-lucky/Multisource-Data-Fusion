import os
import time
import datetime
import os
import shutil
import sys

import torch
from torchvision import transforms
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from model import Deeplab_Torch_Multisource

from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric

from train_and_eval import create_lr_scheduler, validation, save_checkpoint, train_one_epoch


def main(args):
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format('fcn', 'wlkdata', get_rank()))
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    args.base_size = 224
    args.crop_size = args.base_size
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}

    # 加载数据集
    train_dataset = get_segmentation_dataset(args.data_set, root=args.data_dir, split='train', mode='train',
                                             **data_kwargs)

    val_dataset = get_segmentation_dataset(args.data_set, root=args.data_dir, split='val', mode='val',
                                           **data_kwargs)

    iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
    max_iters = args.epochs * iters_per_epoch
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, max_iters)
    val_sampler = make_data_sampler(val_dataset, False, args.distributed)
    val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_sampler=train_batch_sampler,
                                   num_workers=num_workers,
                                   pin_memory=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_sampler=val_batch_sampler,
                                 num_workers=num_workers,
                                 pin_memory=True)

    model = Deeplab_Torch_Multisource(13)
    model.to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            # assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'{AttributeError}'Deeplab_Torch_Multisource' object has no attribute 'paramters'
            print('Resuming training, loading {}...'.format(args.resume))
            model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

    # create criterion
    criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                                      aux_weight=args.aux_weight, ignore_index=-1).to(device)

    params_list = list()
    if hasattr(model, 'pretrained'):
        params_list.append({'params': model.parameters(), 'lr': args.lr})
    if hasattr(model, 'exclusive'):
        for module in model.exclusive:
            params_list.append({'params': getattr(model, module).parameters(), 'lr': args.lr * 10})

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # lr scheduling
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    metric = SegmentationMetric(train_dataset.num_class)

    # 开始训练
    save_to_disk = True
    iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
    max_iters = args.epochs * iters_per_epoch
    log_per_iters, val_per_iters = 10, iters_per_epoch
    save_per_iters = args.save_epoch * iters_per_epoch
    start_time = time.time()
    logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(args.epochs, max_iters))
    model.train()

    for iteration, (images, targets, imgs_feats) in enumerate(train_loader):
        iteration = iteration + 1
        print('iteration is %d' % iteration)
        losses_reduced = train_one_epoch(images, device, imgs_feats, targets, model, criterion, optimizer, lr_scheduler)

        eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % log_per_iters == 0 and save_to_disk:
            logger.info(
                "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                    iteration, max_iters, optimizer.param_groups[0]['lr'], losses_reduced.item(),
                    str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

        if iteration % save_per_iters == 0 and save_to_disk:
            save_checkpoint(model, args, is_best=False)

        if not args.skip_val and iteration % val_per_iters == 0:
            validation(model, metric, val_loader, device, logger, args)
            model.train()

    total_training_time = time.time() - start_time
    total_training_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f}s / it)".format(
            total_training_str, total_training_time / max_iters))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch deeplabv3 training")
    parser.add_argument('--model', type=str, default='fcn')
    parser.add_argument("--data-dir", default="/root/WLKdataset", help="WLKdata root")
    parser.add_argument("--data-set", default="yjs")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=120, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
    parser.add_argument("--num-gpus", default=1)
    parser.add_argument("--distributed", default=False)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--log-dir', default='../runs/logs/', help='Directory for saving checkpoint models')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--use-ohem', type=bool, default=True,  # 提出的模型跑的时候要设置这个
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: vgg16)')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
