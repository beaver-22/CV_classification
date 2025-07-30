import torch
import numpy as np
import json
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import timm

from datasets import create_dataset, create_dataloader
from main import get_model

def evaluate_model(model, testloader, device, num_classes):
    """모델 평가 함수"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Loss 계산
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 확률과 예측값 저장
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Top-1 정확도
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    # Top-5 정확도 계산
    all_probs = np.array(all_probs)
    top5_correct = 0
    for i, target in enumerate(all_targets):
        top5_preds = np.argsort(all_probs[i])[-5:]
        if target in top5_preds:
            top5_correct += 1
    
    results = {
        'test_loss': total_loss / len(testloader),
        'top1_accuracy': correct / total,
        'top5_accuracy': top5_correct / total,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }
    
    return results

def generate_classification_report(targets, predictions, class_names=None):
    """분류 리포트 생성"""
    report = classification_report(
        targets, predictions, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    return report

def plot_confusion_matrix(targets, predictions, class_names=None, save_path=None):
    """Confusion Matrix 시각화"""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(12, 10))
    
    # class_names가 None이면 tick labels를 표시하지 않음
    if class_names is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 메모리 절약을 위해 닫기
    else:
        plt.show()
def main():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--model-name', type=str, required=True, 
                       choices=['resnet50', 'vit_small_patch16_224'])
    parser.add_argument('--dataname', type=str, required=True,
                       choices=['CIFAR10', 'CIFAR100', 'tiny-imagenet-200'])
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--datadir', type=str, default='./datasets')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 이미지 크기 설정
    img_size = 32 if args.dataname in ['CIFAR10', 'CIFAR100'] else 64
    
    # 모델 로드
    model = get_model(args.model_name, args.num_classes, img_size=img_size, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # 데이터셋 로드 (테스트용 기본 augmentation)
    _, testset = create_dataset(datadir=args.datadir, dataname=args.dataname, aug_name='default')
    testloader = create_dataloader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    
    # 평가 실행
    print("🔍 모델 평가 시작...")
    results = evaluate_model(model, testloader, device, args.num_classes)
    
    # 결과 출력
    print(f"\n{'='*50}")
    print(f"📊 평가 결과")
    print(f"{'='*50}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f} ({results['top1_accuracy']*100:.2f}%)")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
    
    # 분류 리포트 생성
    class_report = generate_classification_report(results['targets'], results['predictions'])
    
    # 결과 저장
    eval_results = {
        'test_loss': results['test_loss'],
        'top1_accuracy': results['top1_accuracy'],
        'top5_accuracy': results['top5_accuracy'],
        'classification_report': class_report
    }
    
    # JSON으로 저장
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    # Confusion Matrix 저장
    plot_confusion_matrix(
        results['targets'], 
        results['predictions'],
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    print(f"\n💾 결과가 {args.output_dir}에 저장되었습니다.")

if __name__ == '__main__':
    main()
