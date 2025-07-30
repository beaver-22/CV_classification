import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def generate_comprehensive_report():
    """종합 실험 리포트 생성"""
    
    # 실험 결과 수집
    results = []
    for exp_dir in Path('saved_model').iterdir():
        if exp_dir.is_dir():
            json_path = exp_dir / 'best_results.json'
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                    
                exp_name = exp_dir.name
                model_type = 'ResNet50' if 'resnet50' in exp_name.lower() else 'ViT-S/16'
                pretrained = 'Pre-trained' if 'pretrained' in exp_name else 'Scratch'
                
                results.append({
                    'Model': model_type,
                    'Training': pretrained,
                    'Accuracy': data['best_acc'],
                    'Epoch': data['best_epoch'],
                    'Experiment': exp_name
                })
    
    df = pd.DataFrame(results)
    
    # 성능 비교 테이블 생성
    pivot_table = df.pivot_table(
        values='Accuracy', 
        index='Model', 
        columns='Training', 
        aggfunc='mean'
    )
    
    print("📊 Performance Comparison Table")
    print("=" * 50)
    print(pivot_table.to_string())
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 성능 비교 바 차트
    pivot_table.plot(kind='bar', ax=axes[0,0], rot=0)
    axes[0,0].set_title('Model Performance Comparison')
    axes[0,0].set_ylabel('Accuracy')
    
    # 2. Pre-training 효과
    improvement = pivot_table['Pre-trained'] - pivot_table['Scratch']
    improvement.plot(kind='bar', ax=axes[0,1], color=['skyblue', 'lightcoral'])
    axes[0,1].set_title('Pre-training Improvement')
    axes[0,1].set_ylabel('Accuracy Improvement')
    axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, pivot_table

if __name__ == "__main__":
    results_df, performance_table = generate_comprehensive_report()
