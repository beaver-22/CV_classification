import numpy as np
import os
import random
import wandb

import torch
import argparse
import timm
import logging
from tqdm import tqdm

from train import fit
from models import *
from datasets import create_dataset, create_dataloader
from log import setup_default_logging

_logger = logging.getLogger('train')

# 시드 고정 함수를 통한 랜덤성 제어
def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def get_model(model_name, num_classes, img_size=32, pretrained=False):
    print(f">>> get_model called with pretrained={pretrained}")
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
        return model
    elif model_name == 'vit_small_patch16_224':
        model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes, img_size=img_size)
        return model

'''
def get_model(model_name, num_classes, img_size=32):
    if model_name == 'ResNet18':
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)
    elif model_name == 'ResNet34':
        from models.resnet import ResNet34
        return ResNet34(num_classes=num_classes)
    elif model_name == 'ResNet50':
        from models.resnet import ResNet50
        return ResNet50(num_classes=num_classes)
    elif model_name == 'ResNet101':
        from models.resnet import ResNet101
        return ResNet101(num_classes=num_classes)
    elif model_name == 'ResNet152':
        from models.resnet import ResNet152
        return ResNet152(num_classes=num_classes)
    elif model_name == 'ViT_Tiny':
        from models.vit import ViT_Tiny
        return ViT_Tiny(num_classes=num_classes, img_size=img_size)
    elif model_name == 'ViT_Small':
        from models.vit import ViT_Small
        return ViT_Small(num_classes=num_classes, img_size=img_size)
    elif model_name == 'ViT_Base':
        from models.vit import ViT_Base
        return ViT_Base(num_classes=num_classes, img_size=img_size)
    elif model_name == 'ViT_Large':
        from models.vit import ViT_Large
        return ViT_Large(num_classes=num_classes, img_size=img_size)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
'''

def run(args):
    # make save directory
    savedir = os.path.join(args.savedir, args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # CIFAR10, CIFAR100은 32x32 이미지, TinyImagenet은 64x64 이미지로 가정
    img_size = 32 if args.dataname in ['CIFAR10','CIFAR100'] else 64 

    # augmentation 설정에 따른 MixUp/CutMix 플래그 설정
    use_mixup = args.aug_name == 'mixup'
    use_cutmix = args.aug_name == 'cutmix'

    # --- 모델 생성 부분 개선 ---
    model = get_model(args.model_name, args.num_classes, img_size=img_size, pretrained=args.pretrained)
    model.to(device)
    
    _logger.info('Model: {}'.format(args.model_name))
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # load dataset
    trainset, testset = create_dataset(datadir=args.datadir, dataname=args.dataname, aug_name=args.aug_name)
    
    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=256, shuffle=False)

    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.opt_name](model.parameters(), lr=args.lr)

    # scheduler
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # initialize wandb
    wandb.init(name=args.exp_name, project='CV-study-final', config=args)

    # fitting model
    # train.py 파일의 fit 함수를 호출하여 모델 학습을 시작합니다.
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = args.epochs, 
        savedir      = savedir,
        log_interval = args.log_interval,
        device       = device,
        use_mixup=use_mixup,      
        use_cutmix=use_cutmix,    
        alpha=1.0)                

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Classification for Computer Vision")
    # model
    parser.add_argument('--model-name', type=str, default='ResNet18',
                   choices=['resnet50', 'vit_small_patch16_32', 'vit_small_patch16_224'], #'ResNet18','ResNet34','ResNet50','ViT_Tiny','ViT_Small','ViT_Base','ViT_Large'],
                   help='모델 이름')    
    # pretrained
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained weights')
    
    # exp setting
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--datadir',type=str,default='/datasets',help='data directory')

    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')

    # datasets
    parser.add_argument('--dataname',type=str,default='CIFAR100',choices=['CIFAR10','CIFAR100', 'tiny-imagenet-200'],help='target dataname')
    parser.add_argument('--num-classes',type=int,default=100,help='target classes')

    # optimizer
    parser.add_argument('--opt-name',type=str,choices=['SGD','Adam'],help='optimizer name')
    parser.add_argument('--lr',type=float,default=0.1,help='learning_rate')

    # scheduler "CosineAnnealingLR"로 고정되어 있음. on/off 여부만 설정
    parser.add_argument('--use_scheduler',action='store_true',help='use sheduler')

    # augmentation
    parser.add_argument('--aug-name',type=str,choices=['default','weak','strong', 'cutmix', 'mixup', 'cifar'],help='augmentation type')

    # train
    parser.add_argument('--epochs',type=int,default=50,help='the number of epochs')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')

    # seed
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')

    args = parser.parse_args()

    run(args)