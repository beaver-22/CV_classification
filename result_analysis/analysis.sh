# 2. Attribution 분석
echo "🔍 Phase 2: Attribution Analysis"
python attribution_analysis.py --model-path ../saved_model/RCIFAR100-resnet50-False-aug_cutmix/best_model.pt --model-name resnet50

# 3. Feature 분석
echo "📈 Phase 3: Feature Analysis"
python feature_analysis.py --model-path ../saved_model/RCIFAR100-resnet50-False-aug_cutmix/best_model.pt --model-name resnet50