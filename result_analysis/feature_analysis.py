import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class ModelAnalysis:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_features(self, dataloader, layer_name=None):
        """특정 레이어에서 feature 추출"""
        features = []
        labels = []
        
        # Hook을 통한 intermediate feature 추출
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # 마지막 레이어 전 features 추출
        if hasattr(self.model, 'fc'):  # ResNet
            self.model.avgpool.register_forward_hook(get_activation('features'))
        elif hasattr(self.model, 'head'):  # ViT
            self.model.norm.register_forward_hook(get_activation('features'))
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # Features 추출
                feat = activation['features']
                if len(feat.shape) > 2:  # ResNet의 경우
                    feat = feat.mean(dim=[-2, -1])  # Global average pooling
                elif len(feat.shape) == 3:  # ViT의 경우
                    feat = feat[:, 0, :]  # CLS token
                
                features.append(feat.cpu().numpy())
                labels.append(targets.cpu().numpy())
        
        return np.vstack(features), np.hstack(labels)
    
    def plot_pca(self, features, labels, n_components=2, save_path=None):
        """PCA 시각화"""
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6, s=1)
        plt.colorbar(scatter)
        plt.title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def plot_tsne(self, features, labels, save_path=None):
        """t-SNE 시각화"""
        # 샘플링 (t-SNE는 계산 비용이 높음)
        if len(features) > 5000:
            indices = np.random.choice(len(features), 5000, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]
        else:
            features_sample = features
            labels_sample = labels
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features_sample)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                            c=labels_sample, cmap='tab10', alpha=0.6, s=1)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
