#dataname=CIFAR100
#num_classes=100
dataname=tiny-imagenet-200
num_classes=200


model_list='vit_small_patch16_224'
pretrained_list='True'
aug_list='cutmix' #cifar

lr=0.001
opt=Adam
datadir=./datasets
batch_size=256
epoch=50

# 상위 디렉토리로 이동 (main.py가 있는 폴더)
cd "$(dirname "$0")/.."

for model in $model_list
do
    for pretrained in $pretrained_list
    do
        pretrained_flag=""
        if [ "$pretrained" == "True" ]; then
            pretrained_flag="--pretrained"
        fi
        for aug in $aug_list
        do
            # use scheduler
            echo "----------------------------------"
            echo "OOOO model: $model data: $dataname, pretrain: $pretrained aug: $aug OOOO"
            echo  "----------------------------------"
            EXP_NAME="$dataname-$model-$pretrained-aug_$aug"
            if [ -d "$EXP_NAME" ]
            then
                echo "$EXP_NAME is exist"
            else
                python main.py \
                    --model-name $model \
                    --exp-name $EXP_NAME \
                    --dataname $dataname \
                    --num-classes $num_classes \
                    --opt-name $opt \
                    --aug-name $aug \
                    --batch-size $batch_size \
                    --lr $lr \
                    --use_scheduler \
                    --epochs $epoch \
                    --datadir $datadir \
                    $pretrained_flag
            fi
        done
    done
done