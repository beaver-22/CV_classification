#!/bin/bash
# eval_batch.sh - 여러 모델 일괄 평가

# 평가할 모델들 리스트
MODELS=(
    "CIFAR100-resnet50-False-aug_cutmix"
    "CIFAR100-resnet50-False-aug_default"
)

for model_dir in "${MODELS[@]}"
do
    model_path="./saved_model/$model_dir/best_model.pt"
    
    if [ -f "$model_path" ]; then
        echo "📊 평가 중: $model_dir"
        
        # 모델 타입과 데이터셋 파싱
        if [[ $model_dir == *"vit"* ]]; then
            model_name="vit_small_patch16_224"
        else
            model_name="resnet50"
        fi
        
        if [[ $model_dir == *"CIFAR100"* ]]; then
            dataname="CIFAR100"
            num_classes=100
        else
            dataname="tiny-imagenet-200"
            num_classes=200
        fi
        
        # 평가 실행
        python evaluation.py \
            --model-path "$model_path" \
            --model-name "$model_name" \
            --dataname "$dataname" \
            --num-classes "$num_classes" \
            --output-dir "./evaluation_results/$model_dir"
        
        echo "✅ $model_dir 평가 완료"
        echo "----------------------------------------"
    else
        echo "⚠️  $model_path 파일이 없습니다."
    fi
done

echo "🎉 모든 모델 평가 완료!"
