import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import time


_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """MixUp 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    """CutMix 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # Bounding box 생성
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 중심점 랜덤 선택
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Bounding box 경계 설정
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 실제 비율 재계산
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    # CutMix 적용
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp/CutMix용 손실 함수"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def accuracy(output, target, topk=(1,)):
    """정확도 계산 함수"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(model, dataloader, criterion, optimizer, log_interval: int, device: str, use_mixup=False, use_cutmix=False, alpha=1.0) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    
    # 🔥 tqdm으로 진행률 표시 (배치별 로그 대신)
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for idx, (inputs, targets) in enumerate(pbar):
        data_time_m.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)

        # MixUp 또는 CutMix 적용
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha, use_cuda=(device != 'cpu'))
        elif use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha, use_cuda=(device != 'cpu'))

        optimizer.zero_grad()

        # predict
        outputs = model(inputs)
        
        # 🔥 ViT 모델인 경우 tuple 반환 처리
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        # loss 계산
        if use_mixup or use_cutmix:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
            
        loss.backward()

        # loss update
        optimizer.step()
        losses_m.update(loss.item())

        # accuracy 계산
        if use_mixup or use_cutmix:
            # MixUp/CutMix 시 두 타겟의 가중 평균으로 정확도 계산
            acc1_a = accuracy(outputs, targets_a, topk=(1,))[0]
            acc1_b = accuracy(outputs, targets_b, topk=(1,))[0]
            acc1 = lam * acc1_a + (1 - lam) * acc1_b
            acc_m.update(acc1.item(), n=targets.size(0))
        else:
            preds = outputs.argmax(dim=1) 
            acc_m.update(targets.eq(preds).sum().item()/targets.size(0), n=targets.size(0))
        
        batch_time_m.update(time.time() - end)
        
        # 🔥 tqdm 진행률 업데이트 (매 배치마다 한 줄로 덮어씀)
        pbar.set_postfix({
            'Loss': f'{losses_m.val:.4f}',
            'Acc': f'{acc_m.avg:.3%}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
   
        end = time.time()
    
    return OrderedDict([('acc', acc_m.avg), ('loss', losses_m.avg)])


def test(model, dataloader, criterion, log_interval: int, device: str) -> dict:
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    
    # 🔥 tqdm으로 진행률 표시 (배치별 로그 대신)
    pbar = tqdm(dataloader, desc='Testing', leave=False)
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            
            # 🔥 ViT 모델인 경우 tuple 반환 처리
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # loss 
            loss = criterion(outputs, targets)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += targets.eq(preds).sum().item()
            total += targets.size(0)
            
            # 🔥 tqdm 진행률 업데이트 (매 배치마다 한 줄로 덮어씀)
            pbar.set_postfix({
                'Loss': f'{total_loss/(idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
                
    return OrderedDict([('acc', correct/total), ('loss', total_loss/len(dataloader))])

def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, 
    epochs: int, savedir: str, log_interval: int, device: str,
    use_mixup=False, use_cutmix=False, alpha=1.0  # ← 이 줄 추가
) -> None:

    best_acc = 0
    step = 0
    
    for epoch in range(epochs):
        # 🔥 에폭 시작 로그 (간단하게)
        _logger.info(f'\n{"="*50}')
        _logger.info(f'Epoch {epoch+1:3d}/{epochs}')
        _logger.info(f'{"="*50}')
        
        # 훈련
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, device,
                     use_mixup=use_mixup, use_cutmix=use_cutmix, alpha=alpha)
        
        # 검증
        eval_metrics = test(model, testloader, criterion, log_interval, device)

        # 🔥 에폭별 요약 로그 (한 줄로 깔끔하게)
        current_lr = optimizer.param_groups[0]['lr']
        _logger.info(
            f'Epoch {epoch+1:3d} | '
            f'Train Loss: {train_metrics["loss"]:.4f} Acc: {train_metrics["acc"]:.3%} | '
            f'Val Loss: {eval_metrics["loss"]:.4f} Acc: {eval_metrics["acc"]:.3%} | '
            f'LR: {current_lr:.6f}'
        )

        # wandb 로그 (막대 그래프용)
        metrics = OrderedDict(lr=current_lr)
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        metrics['epoch'] = epoch + 1  # 🔥 에폭 정보 추가 (x축용)
        wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_acc < eval_metrics['acc']:
            # save results
            state = {'best_epoch': epoch, 'best_acc': eval_metrics['acc']}
            json.dump(state, open(os.path.join(savedir, f'best_results.json'), 'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
            
            # 🔥 베스트 모델 저장 로그 (간단하게)
            _logger.info(f'💾 Best model saved! Val Acc: {best_acc:.3%} → {eval_metrics["acc"]:.3%}')

            best_acc = eval_metrics['acc']

    # 🔥 최종 결과 로그
    _logger.info(f'\n{"="*50}')
    _logger.info(f'🎉 Training Completed!')
    _logger.info(f'Best Accuracy: {state["best_acc"]:.3%} (Epoch {state["best_epoch"]+1})')
    _logger.info(f'{"="*50}')
