import math
import torch
from torch import nn


class NewGELUActivation(nn.Module):
    """    
    New GELU 활성화 함수
    출처: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    이미지를 패치로 나누고 벡터 공간으로 투영하는 모듈
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # 이미지 크기와 패치 크기로부터 패치 개수 계산
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 이미지를 패치로 변환하는 투영 레이어 생성
        # 각 패치를 hidden_size 크기의 벡터로 투영
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        # 출력: (batch_size, num_channels, image_size, image_size) -> (batch_size, hidden_size, num_patches_height, num_patches_width)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, hidden_size, num_patches) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    패치 임베딩을 클래스 토큰 및 위치 임베딩과 결합하는 모듈
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # 학습 가능한 [CLS] 토큰 생성
        # BERT와 유사하게 [CLS] 토큰이 입력 시퀀스의 시작에 추가되며
        # 전체 시퀀스를 분류하는 데 사용됨
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # [CLS] 토큰과 패치 임베딩을 위한 위치 임베딩 생성
        # [CLS] 토큰을 위해 시퀀스 길이에 1을 추가
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # [CLS] 토큰을 배치 크기만큼 확장
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # [CLS] 토큰을 입력 시퀀스의 시작에 연결
        # 이로 인해 시퀀스 길이가 (num_patches + 1)이 됨
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    단일 어텐션 헤드
    MultiHeadAttention 모듈에서 사용됨
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # query, key, value 투영 레이어 생성
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 입력을 query, key, value로 투영
        # 동일한 입력으로 query, key, value를 생성하므로
        # 보통 self-attention이라고 부름
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # 어텐션 스코어 계산
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # 어텐션 출력 계산
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    멀티헤드 어텐션 모듈
    TransformerEncoder 모듈에서 사용됨
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # 어텐션 헤드 크기는 hidden size를 어텐션 헤드 개수로 나눈 값
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # query, key, value 투영 레이어에서 bias 사용 여부
        self.qkv_bias = config["qkv_bias"]
        # 어텐션 헤드 리스트 생성
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # 어텐션 출력을 다시 hidden size로 투영하는 선형 레이어 생성
        # 대부분의 경우 all_head_size와 hidden_size는 동일함
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 각 어텐션 헤드에 대한 어텐션 출력 계산
        attention_outputs = [head(x) for head in self.heads]
        # 각 어텐션 헤드의 어텐션 출력을 연결
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # 연결된 어텐션 출력을 다시 hidden size로 투영
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # 어텐션 출력과 어텐션 확률 반환 (선택사항)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    최적화된 멀티헤드 어텐션 모듈
    모든 헤드가 병합된 query, key, value 투영으로 동시에 처리됨
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # 어텐션 헤드 크기는 hidden size를 어텐션 헤드 개수로 나눈 값
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # query, key, value 투영 레이어에서 bias 사용 여부
        self.qkv_bias = config["qkv_bias"]
        # query, key, value를 투영하는 선형 레이어 생성
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # 어텐션 출력을 다시 hidden size로 투영하는 선형 레이어 생성
        # 대부분의 경우 all_head_size와 hidden_size는 동일함
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # query, key, value 투영
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # 투영된 query, key, value를 query, key, value로 분할
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # query, key, value를 (batch_size, num_attention_heads, sequence_length, attention_head_size)로 변형
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # 어텐션 스코어 계산
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # 어텐션 출력 계산
        attention_output = torch.matmul(attention_probs, value)
        # 어텐션 출력 크기 변경
        # (batch_size, num_attention_heads, sequence_length, attention_head_size)에서
        # (batch_size, sequence_length, all_head_size)로
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # 어텐션 출력을 다시 hidden size로 투영
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # 어텐션 출력과 어텐션 확률 반환 (선택사항)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    다중 레이어 퍼셉트론 모듈 (Feed-Forward Network)
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    단일 트랜스포머 블록
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # 잔차 연결 (Skip connection)
        x = x + attention_output
        # Feed-forward 네트워크
        mlp_output = self.mlp(self.layernorm_2(x))
        # 잔차 연결 (Skip connection)
        x = x + mlp_output
        # 트랜스포머 블록의 출력과 어텐션 확률 반환 (선택사항)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    트랜스포머 인코더 모듈
    """

    def __init__(self, config):
        super().__init__()
        # 트랜스포머 블록 리스트 생성
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # 각 블록에 대한 트랜스포머 블록의 출력 계산
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # 인코더의 출력과 어텐션 확률 반환 (선택사항)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    분류를 위한 ViT 모델
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # 임베딩 모듈 생성
        self.embedding = Embeddings(config)
        # 트랜스포머 인코더 모듈 생성
        self.encoder = Encoder(config)
        # 인코더의 출력을 클래스 수로 투영하는 선형 레이어 생성
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # 임베딩 출력 계산
        embedding_output = self.embedding(x)
        # 인코더의 출력 계산
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # 로짓 계산, 분류를 위해 [CLS] 토큰의 출력을 특징으로 사용
        logits = self.classifier(encoder_output[:, 0, :])
        # 로짓과 어텐션 확률 반환 (선택사항)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        """가중치 초기화 함수"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)


# ResNet 스타일의 편의 함수들 추가
def get_vit_config(model_size='base', num_classes=100, img_size=32):
    """ViT 설정을 반환하는 함수"""
    
    base_config = {
        "image_size": img_size,
        "patch_size": 4 if img_size == 32 else 16,
        "num_channels": 3,
        "num_classes": num_classes,
        "qkv_bias": True,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "use_faster_attention": True,  # 최적화된 어텐션 사용
    }
    
    if model_size == 'tiny':
        base_config.update({
            "hidden_size": 192,
            "intermediate_size": 768,
            "num_attention_heads": 3,
            "num_hidden_layers": 12,
        })
    elif model_size == 'small':
        base_config.update({
            "hidden_size": 384,
            "intermediate_size": 1536,
            "num_attention_heads": 6,
            "num_hidden_layers": 12,
        })
    elif model_size == 'base':
        base_config.update({
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
        })
    elif model_size == 'large':
        base_config.update({
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
        })
    
    return base_config


def ViT_Tiny(num_classes=100, img_size=32):
    """ViT-Tiny 모델"""
    config = get_vit_config('tiny', num_classes, img_size)
    return ViTForClassfication(config)


def ViT_Small(num_classes=100, img_size=32):
    """ViT-Small 모델"""
    config = get_vit_config('small', num_classes, img_size)
    return ViTForClassfication(config)


def ViT_Base(num_classes=100, img_size=32):
    """ViT-Base 모델"""
    config = get_vit_config('base', num_classes, img_size)
    return ViTForClassfication(config)


def ViT_Large(num_classes=100, img_size=32):
    """ViT-Large 모델"""
    config = get_vit_config('large', num_classes, img_size)
    return ViTForClassfication(config)

'''
def test():
    """테스트 함수"""
    model = ViT_Tiny(num_classes=100)
    x = torch.randn(1, 3, 32, 32)
    logits, _ = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    test()
'''