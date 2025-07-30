SESSION_NAME="cv_cifar"
echo "🚀 Starting 4 parallel CV experiments in tmux..."

# 기존 세션 종료
tmux kill-session -t $SESSION_NAME 2>/dev/null

tmux new-session -d -s $SESSION_NAME "bash scripts/run_cifar_res.sh"
sleep 0.2
tmux split-window -v -t $SESSION_NAME "bash scripts/run_cifar_res_pre.sh"
sleep 0.2
tmux split-window -h -t $SESSION_NAME:0.0 "bash scripts/run_cifar_vit.sh"
sleep 0.2
tmux split-window -h -t $SESSION_NAME:0.1 "bash scripts/run_cifar_vit_pre.sh"
sleep 0.2

# 창 레이아웃 자동 정렬 (4분할)
tmux select-layout -t $SESSION_NAME tiled

# 팬마다 안내 메시지
tmux select-pane -t $SESSION_NAME:0.0
tmux send-keys "echo 'CIFAR-ResNet 실험 실행 중...'" C-m

tmux select-pane -t $SESSION_NAME:0.1
tmux send-keys "echo 'CIFAR-ViT 실험 실행 중...'" C-m

tmux select-pane -t $SESSION_NAME:0.2
tmux send-keys "echo 'Tiny-ResNet-Pre 실험 실행 중...'" C-m

tmux select-pane -t $SESSION_NAME:0.3
tmux send-keys "echo 'Tiny-ViT-Pre 실험 실행 중...'" C-m

echo "✅ 4 experiments started in tmux session: $SESSION_NAME"
echo "📊 To view progress: tmux attach -t $SESSION_NAME"
echo "❌ To stop all: tmux kill-session -t $SESSION_NAME"

# 세션 자동 접속
tmux attach -t $SESSION_NAME
