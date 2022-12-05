import torch
import time
import os
import sys
sys.path.append("../")

import numpy as np
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.backends import cudnn
from torch.optim import SGD, Adam, lr_scheduler, AdamW

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import AverageMeter, calculate_accuracy, accuracy
from utils import Logger, worker_init_fn, get_lr, freeze_model
from configuration import build_config
from myutils.aj_lr import adjust_learning_rate_contrast, adjust_learning_rate, adjust_learning_rate_cos
from setting import save_checkpoint, save_best, get_model, get_opt
from tqdm import tqdm

"""dataset"""
from DA_dataset_1 import BaseDataset_LR_HR
from utils import EvalLogger

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def train_epoch(epoch,
                data_loader,
                model,
                criterion,  # bce
                bce_criterion,  # cos
                optimizer1,
                device,
                epoch_logger1, epoch_logger2,
                tb_writer=None,
                ):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    bce_losses = AverageMeter()
    contrast_losses = AverageMeter()
    similar_losses = AverageMeter()
    ###
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    con_train_top1 = AverageMeter()
    con_train_top5 = AverageMeter()

    end_time = time.time()
    for i, [clip_lr, clip_hr, label] in enumerate(tqdm(data_loader)):
        data_time.update(time.time() - end_time)

        clip_lr = clip_lr.to(device, non_blocking=True).float()  # 14 x 14
        clip_hr = clip_hr.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).float()

        y_HR, z_HR, y_LR, z_LR = model(clip_lr, clip_hr, mode="train")

        """修改损失函数"""
        """超参数调整"""

        contrast_loss, similar_loss = criterion(y_LR, y_HR)
        bce_loss = bce_criterion(z_LR, label)
        l1, l2, l3 = 1, 0.1, 0.1
        loss = l1 * contrast_loss + l2 * similar_loss + l3 * bce_loss
        if i == 0:
            print("clip_lr:", clip_lr.shape)
            print("clip_hr:", clip_hr.shape)

            print("contrast_loss:", contrast_loss)
            print("similar_loss:", similar_loss)
            print("bce_loss:", bce_loss)
            print("requires_grad:", loss.requires_grad)

        bce_losses.update(bce_loss.item(), label.shape[0])
        contrast_losses.update(contrast_loss.item(), label.shape[0])
        similar_losses.update(similar_loss.item(), label.shape[0])
        losses.update(loss.item(), label.shape[0])

        probs = nn.Softmax(dim=1)(z_LR)
        prec1, prec5 = accuracy(probs, label, topk=(1, 5))
        train_top1.update(prec1.item(), label.shape[0])
        train_top5.update(prec5.item(), label.shape[0])

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    if epoch_logger1 is not None:
        log_info = "epoch={}, train_top1={:.4f}, train_top5={:.4f}, lr={:.6f}\n"\
            .format(epoch, train_top1.avg, train_top5.avg, get_lr(optimizer1))
        loss_info = "bce_losses={:.5f}, contrast_losses={:.5f}, similar_losses={:.5f}, loss={:.5f}\n"\
            .format(bce_losses.avg, contrast_losses.avg, similar_losses.avg, losses.avg)
        epoch_logger1.write(log_info+loss_info)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/train_top1', train_top1.avg, epoch)
        tb_writer.add_scalar('train/train_top5', train_top5.avg, epoch)
        tb_writer.add_scalar('train/lr', get_lr(optimizer1), epoch)


def test_epoch(epoch,
               data_loader,
               model,
               device,
               logger1, logger2,
               tb_writer=None,
               distributed=False):
    print('test at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ###
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    con_test_top1 = AverageMeter()
    con_test_top5 = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, [clip_lr, clip_hr, label] in enumerate(tqdm(data_loader)):
            data_time.update(time.time() - end_time)

            clip_lr = clip_lr.to(device, non_blocking=True).float()  # 14 x 14
            clip_hr = clip_hr.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).float()

            y, z = model(clip_lr, mode="test")
            probs = nn.Softmax(dim=1)(z)
            prec1, prec5 = accuracy(probs, label, topk=(1, 5))
            test_top1.update(prec1.item(), label.shape[0])
            test_top5.update(prec5.item(), label.shape[0])
            ###

            if epoch<3 or epoch>opt.n_epochs-3:
                y1, z1 = model.testHR(clip_hr)
                probs = nn.Softmax(dim=1)(z1)
                prec1, prec5 = accuracy(probs, label, topk=(1, 5))
                con_test_top1.update(prec1.item(), label.shape[0])
                con_test_top5.update(prec5.item(), label.shape[0])

            batch_time.update(time.time() - end_time)
            end_time = time.time()

    if logger1 is not None:
        log_info = "epoch={}, test_top1={:.4f}, test_top5={:.4f}\n".format(epoch, test_top1.avg, test_top5.avg)
        logger1.write(log_info)
    if logger2 is not None:
        log_info = "epoch={}, con_test_top1={:.4f}, con_test_top5={:.4f}\n".format(epoch, con_test_top1.avg, con_test_top5.avg)
        logger2.write(log_info)

    """save best"""
    save_best(opt, model, test_top1, test_top5, epoch, name="rgb_best.pth")
    """end"""

    if tb_writer is not None:
        tb_writer.add_scalar('test/test_top1', test_top1.avg, epoch)
        tb_writer.add_scalar('test/test_top5', test_top5.avg, epoch)


def resume_model(resume_path, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    # assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def get_train_utils(opt, cfg, model_parameters1):
    # Get training data
    train_data = BaseDataset_LR_HR(cfg=cfg, split='train', num_frame=16, lr_size=opt.sample_size, hr_size=opt.hr_size,
                                        resolution="LR")

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads,
                                               drop_last=True)
    # pin_memory=True,
    # sampler=train_sampler,
    # worker_init_fn=worker_init_fn)

    train_logger_LR = EvalLogger(opt.result_path / 'train_lr.log')
    train_logger_HR = EvalLogger(opt.result_path / 'train_hr.log')

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening

    if opt.optimizer == 'sgd':
        # optimizer = SGD(model_parameters,
        #                 lr=opt.learning_rate,
        #                 momentum=opt.momentum,
        #                 dampening=dampening,
        #                 weight_decay=opt.weight_decay,
        #                 nesterov=opt.nesterov)
        optimizer1 = SGD(model_parameters1,
                         opt.learning_rate,
                         momentum=opt.momentum,
                         weight_decay=opt.weight_decay)

    elif opt.optimizer == 'adam':
        optimizer1 = AdamW(model_parameters1,
                           lr=opt.learning_rate,
                           weight_decay=opt.weight_decay,
                           eps=1e-8)
    else:
        print("=" * 40)
        print("Invalid optimizer mode: ", opt.optimizer)
        print("Select [sgd, adam]")
        exit(0)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)

    return (train_loader, train_logger_LR, train_logger_HR, optimizer1)


def get_test_utils(opt, cfg):
    # Get validation data
    test_data = BaseDataset_LR_HR(cfg=cfg, split='test', num_frame=16, lr_size=opt.sample_size, hr_size=opt.hr_size, resolution="LR")
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=opt.n_threads)
    # pin_memory=True,
    # sampler=val_sampler,
    # worker_init_fn=worker_init_fn)

    test_logger_LR = EvalLogger(opt.result_path / 'test_lr.log')
    test_logger_HR = EvalLogger(opt.result_path / 'test_hr.log')

    return test_loader, test_logger_LR, test_logger_HR


def main_worker(opt):
    """

    :param opt:
    :return:
    """

    """设置随机种子"""
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    """"""

    cfg = build_config(opt.dataset)

    """对比训练新的模型， my method"""
    from model_28_56 import myNet
    model = myNet(num_classes=opt.n_classes, hr_size=opt.hr_size)
    freeze_model(model.fix_backbone)
    # freeze_model(model.fc)
    model.to(opt.device)

    parameters = [
        {'params': model.backbone1.parameters(), 'fix_lr': False},
        {'params': model.predictor.parameters(), 'fix_lr': False},
        {'params': model.fc.parameters(), 'fix_lr': 0.01},
    ]  # fix_lr使得 classifier的学习率为原来的0.1

    # 设置损失函数
    from model_28_56 import myLossV1
    criterion = myLossV1().to(opt.device)
    bce_criterion = nn.BCEWithLogitsLoss().to(opt.device)

    if not opt.no_train:
        (train_loader, train_logger1, train_logger2, optimizer1) = get_train_utils(opt, cfg, parameters)

    if not opt.no_test:
        test_loader, test_logger1, test_logger2 = get_test_utils(opt, cfg)

    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            adjust_learning_rate_cos(optimizer1, opt.learning_rate, epoch=i, end_epoch=opt.n_epochs)

            train_epoch(i, train_loader,
                        model,
                        criterion,
                        bce_criterion,
                        optimizer1,
                        opt.device,
                        train_logger1, train_logger2,
                        tb_writer)

            if i % opt.checkpoint == 0 or i % opt.n_epochs == 0:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, model, optimizer1)

        if not opt.no_test:
            test_epoch(i, test_loader,
                       model,
                       opt.device,
                       test_logger1, test_logger2,
                       tb_writer)


if __name__ == '__main__':

    opt = get_opt()

    if not opt.no_cuda:
        cudnn.benchmark = True

    main_worker(opt)
