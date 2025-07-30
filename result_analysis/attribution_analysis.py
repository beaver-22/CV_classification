import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class VanillaGradients:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def generate_gradients(self, input_image, target_class):
        """Vanilla gradients를 통한 attribution 생성"""
        input_image.requires_grad_()
        
        # Forward pass
        output = self.model(input_image)
        
        # Target class에 대한 gradient 계산
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Input gradient 추출
        gradients = input_image.grad.data
        
        return gradients
    
    def visualize_attribution(self, image, gradients, save_path='.\attribution_map\attribution_map.png'):
        """Attribution 시각화"""
        # Gradient 절댓값의 최대값으로 normalize
        attr = torch.abs(gradients).max(dim=1, keepdim=True)[0]
        attr = (attr - attr.min()) / (attr.max() - attr.min())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 원본 이미지
        axes[0].imshow(image.squeeze().permute(1, 2, 0))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attribution map
        axes[1].imshow(attr.squeeze(), cmap='hot')
        axes[1].set_title('Attribution Map')
        axes[1].axis('off')
        
        # Overlay
        overlay = image.squeeze().permute(1, 2, 0) * 0.7 + attr.squeeze().unsqueeze(2) * 0.3
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
