from transformers import PretrainedConfig


class MyMindConfig(PretrainedConfig):
    """
    Configuration class for MyMind model.
    Args:
        dropout (float): Dropout probability.
        bos_token_id (int): Beginning of sequence token ID.
        eos_token_id (int): End of sequence token ID.
        hidden_act (str): Activation function.
        hidden_size (int): Size of hidden layers.
        intermediate_size (int): Size of intermediate layers.
        max_position_embeddings (int): Maximum position embeddings.
        num_attention_heads (int): Number of attention heads.
        num_hidden_layers (int): Number of hidden layers.
        num_key_value_heads (int): Number of key-value heads.
        vocab_size (int): Vocabulary size.
        rms_norm_eps (float): RMSNorm epsilon.
        rope_theta (int): RoPE base frequency.
        inference_rope_scaling (bool): Whether to use RoPE scaling during inference.
        flash_attention (bool): Whether to use flash attention during training.
        use_moe (bool): Whether to use Mixture of Experts.
        num_experts_per_tok (int): Number of experts per token.
        n_routed_experts (int): Number of routed experts.
        n_shared_experts (int): Number of shared experts.
        scoring_func (str): Scoring function for MoE.
        aux_loss_alpha (float): Auxiliary loss weight for MoE.
        seq_aux (bool): Whether to use sequence-level auxiliary loss.
        norm_topk_prob (bool): Whether to normalize top-k probabilities in MoE.
    """

    model_type = "mymind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,    # RMSNorm epsilon
        rope_theta: int = 1000000,      # RoPE base frequency
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,   # Use flash attention during training
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn

# RMSNorm implementation,继承自nn.Module
class RMSNorm(nn.Module):
    # 1. 定义init方法并初始化参数
    def __init__(self, dim:int, eps:float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 gamma,维度与特征通道数相同

    # 2. 计算均方根RMS并归一化
    def _norm(self,x:torch.Tensor):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True)+self.eps)
        return x / rms

    # 3. 定义前向传播方法
    def forward(self, x:torch.Tensor):
        output = self._norm(x) * self.gamma
        return output