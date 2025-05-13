import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('--model-file', type=str, default='Domain3/checkpoint_200.pth.tar')
parser.add_argument('--model', type=str, default='Deeplab', help='Deeplab')
parser.add_argument('--out-stride', type=int, default=16)
parser.add_argument('--sync-bn', type=bool, default=True)
parser.add_argument('--freeze-bn', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr-decrease-rate', type=float, default=0.9, help='ratio multiplied to initial lr')
parser.add_argument('--lr-decrease-epoch', type=int, default=1, help='interval epoch number for lr decrease')

parser.add_argument('--data-dir', default='/Fundus')
parser.add_argument('--datasetT', type=str, default='Domain3')
parser.add_argument('--datasetS', type=str, default='Domain3')
# parser.add_argument('--model-source', type=str, default='Domain2')
parser.add_argument('--batch-size', type=int, default=8)

parser.add_argument('--model-ema-rate', type=float, default=0.98)
parser.add_argument('--pseudo-label-threshold', type=float, default=0.75)
parser.add_argument('--mean-loss-calc-bound-ratio', type=float, default=0.2)

parser.add_argument('--grad_mask_ratio', default=0.3, type=float)
parser.add_argument('--loss_weight_sum', default=0, type=float)

args = parser.parse_args()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import os.path as osp

import numpy as np
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from dataloaders import fundus_dataloader
from dataloaders import custom_transforms as trans
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import networks.GBnet as GBnet
import cv2
import torch.backends.cudnn as cudnn
import random
import glob
import sys
import torch.nn as nn
import gc
from torch.autograd import grad


def snd_loss(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.permute(0,2,3,1).reshape(b,h*w,c)
    p = torch.matmul(x, x.permute(0,2,1))
    p = F.softmax(p, -1)
    e = entropy_loss(p).sum(-1)
    return e.reshape(b, h, w)

def ce_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(y.detach()*(x+1e-30).log())

# def bce_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     return -(y.detach()*(x+1e-30).log() + (1-y.detach())*((1-x+1e-30).log()))

bceloss = torch.nn.BCELoss()

def entropy_loss(x: torch.Tensor) -> torch.Tensor:
    return -(x*(x+1e-30).log() + (1-x)*((1-x+1e-30).log()))

seed = 42
savefig = False
get_hd = True
model_save = True
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def soft_label_to_hard(soft_pls, pseudo_label_threshold):
    pseudo_labels = torch.zeros(soft_pls.size())
    if torch.cuda.is_available():
        pseudo_labels = pseudo_labels.cuda()
    pseudo_labels[soft_pls > pseudo_label_threshold] = 1
    pseudo_labels[soft_pls <= pseudo_label_threshold] = 0

    return pseudo_labels


def init_feature_pred_bank(model, loader):
    feature_bank = {}
    pred_bank = {}
    pred_sum_bank = {}

    pred_sum_bank["epoch"] = 1

    model.eval()

    with torch.no_grad():
        for sample in loader:
            data = sample['image']
            img_name = sample['img_name']
            data = data.cuda()

            pred, _, feat = model(data)
            pred = torch.sigmoid(pred)

            pseudo_labels = soft_label_to_hard(pred, args.pseudo_label_threshold)

            for i in range(data.size(0)):
                feature_bank[img_name[i]] = feat[i].detach().clone()
                pred_bank[img_name[i]] = pred[i].detach().clone()
                pred_sum_bank[img_name[i]] = pseudo_labels[i].detach().clone()

    model.train()

    return feature_bank, pred_bank, pred_sum_bank

from collections import defaultdict, OrderedDict
def get_grads(loss, optimizer, model):
        optimizer.zero_grad()
        loss.backward(inputs=list(model.parameters()),
                          retain_graph=True, create_graph=True)
        dict = OrderedDict(
            [
                (name, weights.grad.clone().view(weights.grad.size(0),-1))
                for name, weights in model.named_parameters()
            ]
        )

        return dict

def l2_between_dicts(dict_1, dict_2, normalize=False):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    dict_1_tensor = torch.cat(tuple([t.view(-1) for t in dict_1_values]))
    dict_2_tensor = torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    if normalize:
        dict_1_tensor = (dict_1_tensor-dict_1_tensor.mean().item()) / dict_1_tensor.std().item()
        dict_2_tensor = (dict_2_tensor-dict_2_tensor.mean().item()) / dict_2_tensor.std().item()
        dict_2_tensor = dict_2_tensor.detach()
    return (dict_1_tensor-dict_2_tensor).pow(2).mean()



def adapt_epoch(model_t, model_s, optim, model_gb, optim_gb, train_loader, args):
    for sample_w, sample_s in train_loader:
        imgs_w = sample_w['image']
        imgs_s = sample_s['image']
        img_name = sample_w['img_name']
        target_map = sample_s['label']
        if torch.cuda.is_available():
            imgs_w = imgs_w.cuda()
            imgs_s = imgs_s.cuda()
            target_map = target_map.cuda()

        # model predict
        predictions_stu_s, _, features_stu_s = model_s(imgs_s)
        with torch.no_grad():
            predictions_tea_w, _, features_tea_w = model_t(imgs_w)

        predictions_stu_s_sigmoid = torch.sigmoid(predictions_stu_s)
        predictions_tea_w_sigmoid = torch.sigmoid(predictions_tea_w)

        pw = model_gb(predictions_stu_s_sigmoid)
        loss_gb = F.mse_loss(pw, predictions_tea_w_sigmoid)

        # get hard pseudo label
        pseudo_labels = soft_label_to_hard(predictions_tea_w_sigmoid, args.pseudo_label_threshold)

        nnbceloss = torch.nn.BCELoss(reduction='none')
        # loss_seg_pixel = nnbceloss(predictions_stu_s_sigmoid, pseudo_labels)
        loss_seg_pixel = bceloss(predictions_stu_s_sigmoid, pseudo_labels)

        loss_seg = torch.mean(loss_seg_pixel)

        ######### update model_s ########
        loss = loss_seg + 0.2*loss_gb
        optim.zero_grad()
        loss.backward()
        optim.step()

        ######### update model_gb ########
        imgs_s = torch.autograd.Variable(imgs_s, requires_grad=True)
        predictions_stu_s, _, features_stu_s = model_s(imgs_s)
        with torch.no_grad():
            predictions_tea_w, _, features_tea_w = model_t(imgs_w)

        predictions_stu_s_sigmoid = torch.sigmoid(predictions_stu_s)
        predictions_tea_w_sigmoid = torch.sigmoid(predictions_tea_w)

        pw = model_gb(predictions_stu_s_sigmoid)
        loss_gb = F.mse_loss(pw, predictions_tea_w_sigmoid, reduction='none')

        # get hard pseudo label
        pseudo_labels = soft_label_to_hard(predictions_tea_w_sigmoid, args.pseudo_label_threshold)

        loss_seg_pixel = bceloss(predictions_stu_s_sigmoid, pseudo_labels)
        # loss_seg = torch.mean(loss_seg_pixel)
        # loss_gb = torch.mean(loss_gb)
        loss = loss_seg_pixel + 0.2*loss_gb

        loss_sup = bceloss(predictions_stu_s_sigmoid, target_map)

        loss_sup0 = torch.mean(loss_sup)
        loss0 = torch.mean(loss)

        dict_sup = get_grads(loss_sup0, optim, model_s.decoder)
        dict_reg = get_grads(loss0, optim, model_s.decoder)


        penalty = l2_between_dicts(dict_reg, dict_sup, normalize=True) * 0.1
        optim_gb.zero_grad()
        penalty.backward(inputs=list(model_gb.parameters()))
        optim_gb.step()

        for param in model_s.decoder.parameters():
            if param.grad is not None:
                param.grad.to('cpu')
                param.to('cpu')
                param.grad = None
                del param.grad

        loss_sup.to('cpu')
        loss_sup = None
        del loss_sup
        loss.to('cpu')
        loss = None
        del loss
        gc.collect(generation=2)
        torch.cuda.empty_cache()

        # update teacher
        for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
            param_t.data = param_t.data.clone() * args.model_ema_rate + param_s.data.clone() * (1.0 - args.model_ema_rate)
    
    print("=============================")




def eval(model, data_loader):
    model.eval()

    val_dice = {'cup': np.array([]), 'disc': np.array([])}
    val_assd = {'cup': np.array([]), 'disc': np.array([])}

    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data = sample['image']
            target_map = sample['label']
            data = data.cuda()
            predictions, _, features = model(data)


            pseudo_label = predictions.clone()
            pseudo_label[pseudo_label > 0.75] = 1.0; pseudo_label[pseudo_label <= 0.75] = 0.0
            target_0_obj = F.interpolate(target_map[:,0:1,...], size=features.size()[2:], mode='nearest')
            target_1_obj = F.interpolate(target_map[:, 1:, ...], size=features.size()[2:], mode='nearest')


            dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)
            val_dice['cup'] = np.append(val_dice['cup'], dice_cup)
            val_dice['disc'] = np.append(val_dice['disc'], dice_disc)

            assd = assd_compute(predictions, target_map)
            val_assd['cup'] = np.append(val_assd['cup'], assd[:, 0])
            val_assd['disc'] = np.append(val_assd['disc'], assd[:, 1])

        avg_dice = [0.0, 0.0, 0.0, 0.0]
        std_dice = [0.0, 0.0, 0.0, 0.0]
        avg_assd = [0.0, 0.0, 0.0, 0.0]
        std_assd = [0.0, 0.0, 0.0, 0.0]
        avg_dice[0] = np.mean(val_dice['cup'])
        avg_dice[1] = np.mean(val_dice['disc'])
        std_dice[0] = np.std(val_dice['cup'])
        std_dice[1] = np.std(val_dice['disc'])
        val_assd['cup'] = np.delete(val_assd['cup'], np.where(val_assd['cup'] == -1))
        val_assd['disc'] = np.delete(val_assd['disc'], np.where(val_assd['disc'] == -1))
        avg_assd[0] = np.mean(val_assd['cup'])
        avg_assd[1] = np.mean(val_assd['disc'])
        std_assd[0] = np.std(val_assd['cup'])
        std_assd[1] = np.std(val_assd['disc'])

    model.train()

    return avg_dice, std_dice, avg_assd, std_assd


def main():
    now = datetime.now()
    here = osp.dirname(osp.abspath(__file__))
    args.out = osp.join(here, 'logs_target', args.datasetT, now.strftime('%Y%m%d_%H%M%S.%f'))
    if not osp.exists(args.out):
        os.makedirs(args.out)
    args.out_file = open(osp.join(args.out, now.strftime('%Y%m%d_%H%M%S.%f')+'.txt'), 'w')
    args.out_file.write(' '.join(sys.argv) + '\n')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    # dataset
    composed_transforms_train = transforms.Compose([
        trans.Resize(512),
        trans.add_salt_pepper_noise(),
        trans.adjust_light(),
        trans.eraser(),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        trans.Resize(512),
        trans.Normalize_tf(),
        trans.ToTensor()
    ])

    dataset_train = fundus_dataloader.FundusSegmentation_2transform(base_dir=args.data_dir, dataset=args.datasetS,
                                                                    split='train/ROIs',
                                                                    transform_weak=composed_transforms_test,
                                                                    transform_strong=composed_transforms_train)
    dataset_test = fundus_dataloader.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='test/ROIs',
                                         transform=composed_transforms_test)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # model
    model_s = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)
    model_t = netd.DeepLab(num_classes=2, backbone='mobilenet', output_stride=args.out_stride, sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)


    if torch.cuda.is_available():
        model_s = model_s.cuda()
        model_t = model_t.cuda()

    log_str = '==> Loading %s model file: %s' % (model_s.__class__.__name__, args.model_file)
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    checkpoint = torch.load(args.model_file)
    model_s.load_state_dict(checkpoint['model_state_dict'])
    model_t.load_state_dict(checkpoint['model_state_dict'])

    if (args.gpu).find(',') != -1:
        model_s = torch.nn.DataParallel(model_s, device_ids=[0, 1])
        model_t = torch.nn.DataParallel(model_t, device_ids=[0, 1])

    optim = torch.optim.Adam(model_s.parameters(), lr=args.lr, betas=(0.9, 0.99))

    model_gb = GBnet.GBnet(sync_bn=args.sync_bn).cuda()
    optim_gb = torch.optim.Adam(model_gb.parameters(), lr=args.lr, betas=(0.9, 0.99))

    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_decrease_epoch, gamma=args.lr_decrease_rate)

    model_s.train()
    model_t.train()
    for param in model_t.parameters():
        param.requires_grad = False


    avg_dice, std_dice, avg_assd, std_assd = eval(model_t, test_loader)
    log_str = ("initial dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
            avg_dice[0], std_dice[0], avg_dice[1], std_dice[1], (avg_dice[0] + avg_dice[1]) / 2.0,
            avg_assd[0], std_assd[0], avg_assd[1], std_assd[1], (avg_assd[0] + avg_assd[1]) / 2.0))
    print(log_str)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    for epoch in range(args.epoch):
        log_str = '\nepoch {}/{}:'.format(epoch+1, args.epoch)
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        adapt_epoch(model_t, model_s, optim, model_gb, optim_gb, train_loader, args)

        scheduler.step()

        avg_dice, std_dice, avg_assd, std_assd = eval(model_t, test_loader)
        log_str = ("teacher dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
            avg_dice[0], std_dice[0], avg_dice[1], std_dice[1], (avg_dice[0] + avg_dice[1]) / 2.0,
            avg_assd[0], std_assd[0], avg_assd[1], std_assd[1], (avg_assd[0] + avg_assd[1]) / 2.0))
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        avg_dice, std_dice, avg_assd, std_assd = eval(model_s, test_loader)
        log_str = ("student dice: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f, assd: cup: %.4f+-%.4f disc: %.4f+-%.4f avg: %.4f" % (
                avg_dice[0], std_dice[0], avg_dice[1], std_dice[1], (avg_dice[0] + avg_dice[1]) / 2.0,
                avg_assd[0], std_assd[0], avg_assd[1], std_assd[1], (avg_assd[0] + avg_assd[1]) / 2.0))
        print(log_str)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()

        torch.save({'model_state_dict': model_gb.state_dict()}, "/model_gb_checkpoint_" + str(epoch+1) + ".pth.tar")

if __name__ == '__main__':
    main()

