import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
# import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image

# from visdom import Visdom

_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)

from torch.utils import data
from tqdm import tqdm

from data import create_dataset
from models import create_model
from utils.utils import get_logger
from augmentations import get_composed_augmentations
from models.adaptation_model import CustomModel
from optimizers import get_optimizer
from schedulers import get_scheduler
from metrics import runningScore, averageMeter
from loss import get_loss_function
from utils import sync_batchnorm
from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def test(cfg, writer, logger):
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))
    ## create dataset
    default_gpu = cfg['model']['default_gpu']
    device = torch.device("cuda:{}".format(default_gpu) if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(cfg, writer, logger)  # source_train\ target_train\ source_valid\ target_valid + _loader

    model = CustomModel(cfg, writer, logger)
    running_metrics_val = runningScore(cfg['data']['target']['n_class'])
    source_running_metrics_val = runningScore(cfg['data']['target']['n_class'])
    val_loss_meter = averageMeter()
    source_val_loss_meter = averageMeter()
    time_meter = averageMeter()
    loss_fn = get_loss_function(cfg)
    path = cfg['test']['path']
    checkpoint = torch.load(path)
    model.adaptive_load_nets(model.BaseNet, checkpoint['DeepLab']['model_state'])
    model.eval(logger=logger)
    validation(
        model, logger, writer, datasets, device, running_metrics_val, val_loss_meter, loss_fn, \
        source_val_loss_meter, source_running_metrics_val, iters=model.iter
    )

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_color_pallete(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'city':
        cityspallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
        #         palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250,
        #         170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
        #         0, 0, 70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        #         zero_pad = 256 * 3 - len(palette)
        #         for i in range(zero_pad):
        #             palette.append(0)
        out_img.putpalette(cityspallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


def validation(model, logger, writer, datasets, device, running_metrics_val, val_loss_meter, loss_fn, \
               source_val_loss_meter, source_running_metrics_val, iters):
    iters = iters
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(
            datasets.target_valid_loader, device, model, running_metrics_val,
            val_loss_meter, loss_fn
        )

    writer.add_scalar('loss/val_loss', val_loss_meter.avg, iters + 1)
    logger.info("Iter %d Loss: %.4f" % (iters + 1, val_loss_meter.avg))

    writer.add_scalar('loss/source_val_loss', source_val_loss_meter.avg, iters + 1)
    logger.info("Iter %d Source Loss: %.4f" % (iters + 1, source_val_loss_meter.avg))

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))
        writer.add_scalar('val_metrics/{}'.format(k), v, iters + 1)

    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))
        writer.add_scalar('val_metrics/cls_{}'.format(k), v, iters + 1)

    val_loss_meter.reset()
    running_metrics_val.reset()

    source_val_loss_meter.reset()
    source_running_metrics_val.reset()

    torch.cuda.empty_cache()
    return score["Mean IoU : \t"]

def validate(valid_loader, device, model, running_metrics_val, val_loss_meter, loss_fn):
    for batch in tqdm(valid_loader):
        images_val = batch['img'].to(device)
        labels_val = batch['label'].to(device)
        outs = model.forward(images_val)
        outputs = F.interpolate(outs['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)
        val_loss = loss_fn(input=outputs, target=labels_val)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)
        val_loss_meter.update(val_loss.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default='configs/test_from_gta_to_city.yml',
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    run_id = random.randint(1, 100000)
    # path = cfg['training']['save_path']
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    # train(cfg, writer, logger)
    test(cfg, writer, logger)