from transformers import PretrainedConfig
import math

# 创建配置对象，后续其他类可以在参数中传入该配置对象以获取超参数
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
import torch.nn.functional as F
from typing import Optional, Tuple

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
    

# Positional Encoding with RoPE 方式1，采用原RoPE论文中的窗口实现
def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e5,
    rope_scaling: Optional[dict] = None,
):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim)) # inverse frequencies

    # YaRN RoPE scaling
    if rope_scaling is not None:
        original_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )

        # 若结束位置超过原始最大位置L_train，则应用YaRN缩放
        if end / original_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > original_max),dim // 2)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)

            # 窗口保留比例beta
            beta = beta_slow + (beta_fast - beta_slow) * power

            # 平滑窗口函数1 按照原论文公式: lambda = (1 + beta*alpha - beta) / (beta*alpha)
            scale = torch.where(
                torch.arange(dim // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor),
                1.0 / factor,
            )
    
            # 应用旋转频率缩放
            freqs = freqs * scale

        m = torch.arange(end, device=freqs.device) # 位置索引
        angles = torch.outer(m, freqs).float() # 外积 m * w_i ，生成旋转角度矩阵: (end, d/2)

        # 生成 cos/sin，并重复每个值两次以匹配嵌入维度
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (end, dim)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)

        return cos, sin

# Positional Encoding with RoPE 方式2，采用主流的简化版公式
def precompute_freqs_cis_simplified(
    dim: int,
    end: int,
    rope_base: float = 10000.0,
    scaling_factor: float = 1.0,   # alpha = L_target / L_train
    beta: float = 0.1,            # 高频保留比例 (e.g., 0.1 = 保留前 10% 高频)
):
    """
    简化版 YaRN：使用单一 beta 控制窗口边界。
    
    Args:
        dim: 嵌入维度（必须为偶数）
        end: 最大位置索引（L_target）
        rope_base: RoPE 基底 theta
        scaling_factor: 扩展因子 alpha
        beta: 高频保留比例，范围 (0, 1]
    returns:
        cos: 位置编码的余弦部分，形状为 (end, dim)
        sin: 位置编码的正弦部分，形状为 (end, dim)
    """
    # 1. 计算原始频率 omega_i = theta^(-2i/d)
    i = torch.arange(0, dim, 2, dtype=torch.float32)  # [0, 2, 4, ..., d-2]
    freqs = 1.0 / (rope_base ** (i / dim))           # shape: (d//2,)

    # 2. 应用 YaRN 缩放（仅当需要外推时）
    if scaling_factor > 1.0:
        half_dim = dim // 2
        # 窗口边界：前 beta * half_dim 个维度为高频保护区
        i_high = max(1, int(beta * half_dim))  # 至少保留1个维度
        
        # 构建窗口函数 w(i) ∈ [0, 1]
        i_idx = torch.arange(half_dim, dtype=torch.float32)
        w = torch.clamp(i_idx / i_high, 0.0, 1.0)  # 线性过渡
        
        # 缩放因子简化公式： lambda_i = 1 + w(i) * (alpha - 1)
        lambda_i = 1.0 + w * (scaling_factor - 1.0)
        
        # 调整频率：omega_i_new = omega_i / lambda_i
        freqs = freqs / lambda_i

    # 3. 生成角度矩阵: (end, d//2)
    m = torch.arange(end, dtype=torch.float32) # 位置索引
    angles = torch.outer(m, freqs)  # m * omega_i 生成旋转角度矩阵

    # 4. 生成 cos/sin，并重复每个值两次以匹配嵌入维度
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (end, dim)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)

    return cos, sin

# 对query和key应用RoPE位置编码
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 构造"垂直向量"
    def rotate_half(x):
        """
        构造“垂直”向量：
        Steps:
            1. 分离向量中的奇数，偶数维度
            2. 将 奇数维度 取负，逐个插入到偶数维度前面, 并将这两个维度展平到一维
        """
        x_even = x[..., ::2] # x0, x2, x4, ... 
        x_odd = x[..., 1::2] # x1, x3, x5, ...
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2) # 在最后一维度堆叠并按照最后两个维度展平
        # ex: [-x1, x0, -x3, x2, -x5, x4]

    q_r = rotate_half(q)
    k_r = rotate_half(k)
    
    # 使用unsqueeze_dim参数在指定维度上扩展cos和sin张量
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (q_r * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (k_r * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv_heads(x:torch.Tensor, num_rep:int):
    """
    重复key/value头以匹配多头注意力中的头数。
    Args:
        x: 输入张量，形状为 (batch_size, seq_len, num_key_value_heads, head_dim)
        num_rep: 重复次数
    Returns:
        重复后的张量，形状为 (batch_size, seq_len, num_key_value_heads * num_rep, head_dim)
    """
    if num_rep==1:
        return x

    # 将 x 在第2维上重复 num_rep 次
    return torch.repeat_interleave(x, repeats=num_rep, dim=2)   # 也可以使用exapnd和reshape实现


# Class GQA
class Attention(nn.Module):
    # config
    def __init__(self, config:MyMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        assert config.num_attention_heads % config.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"
        
        self.num_rep = config.num_attention_heads // config.num_key_value_heads # 计算重复次数       
        self.head_dim = config.hidden_size // config.num_attention_heads # 计算每个头的维度 = 隐藏层大小 / 注意力头数
        
        # 定义投影层. 无偏置，节省参数
        self.q_proj = nn.Linear(config.hidden_size, self.head_dim*config.num_attention_heads, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim*config.num_key_value_heads, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim*config.num_key_value_heads, bias=False)
        self.o_proj = nn.Linear(self.head_dim*config.num_attention_heads, config.hidden_size, bias=False)
        
        # 定义Dropout层
        self.attn_dropout = nn.Dropout(config.dropout) # 在softmax后应用
        self.residual_dropout = nn.Dropout(config.dropout) # 在o_proj后应用与输出
        self.dropout = config.dropout # 存储dropout概率

        # 检测是否支持 Flash Attention 标志
        self.flash=(hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attention)
    

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
    
        batch_size,seq_len, hidden_size = x.shape
        
        # 0. 生成QKV
        q = self.q_proj(x) # (B, L, num_q_heads * head_dim)
        k = self.k_proj(x) # (B, L, num_kv_heads * head_dim)
        v = self.v_proj(x) # (B, L, num_kv_heads * head_dim)
        
        # 1. 分头 
        # 按照 hidden = num_attention_heads * head_dim 对Q分头
        q = q.view(batch_size,seq_len,self.num_attention_heads,self.head_dim)
        # 按照num_key_value_heads * head_dim对KV分头
        k = k.view(batch_size,seq_len,self.num_key_value_heads,self.head_dim)
        v = v.view(batch_size,seq_len,self.num_key_value_heads,self.head_dim)
        
        # 2. 对Q,K使用RoPE位置编码
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])


        # 3.KV cache处理 拼接缓存的past_key和past_value
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)  # 在 seq_len 维度上拼接
            v = torch.cat([past_v, v], dim=1)
        past_kv = (k, v) if use_cache else None # 用于返回缓存的KV

        # 4  repeat kv
        q= q.transpose(1,2)  # (B, num_attention_heads, L, head_dim)    
        k = repeat_kv_heads(k, self.num_rep).transpose(1,2)  # (B, num_attention_heads, L, head_dim)
        v = repeat_kv_heads(v, self.num_rep).transpose(1,2)  # (B, num_attention_heads, L, head_dim)  
        # 转置是为了适配后续的注意力计算 （标准注意力计算的输入要求是(batch, num_heads, seq_len, head_dim)）

        # 5. 注意力计算： 1. 调用Flash Attention 2. 标准注意力计算
        if (
            self.flash
            and seq_len > 1
            and (attention_mask is None or torch.all(attention_mask == 1)) # # 全1表示无mask
        ): # 检查是否支持使用Flash Attention
            
            #设置mask
            attn_mask = (
                None 
                if attention_mask is None 
                else attention_mask.view(batch_size, 1, 1, -1).expand(batch_size, self.num_attention_heads, seq_len, -1).to(torch.bool)
            )

            # 调用内置的点积缩放注意力函数
            output = F.scaled_dot_product_attention(
                q,k,v,attn_mask, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True # 自回归模型
            ) 
        else:
            # 标准注意力计算 q*k^T / sqrt(d)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 计算注意力分数 (B, num_heads, L, L)

            # causal masking: 应用 上三角掩码-inf 以实现自回归 
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))  # 值为0的位置设为-inf

            # 计算注意力权重
            attn_probs = F.softmax(attn_scores, dim=-1)  # (B, num_heads, L, L)
            attn_probs = self.attn_dropout(attn_probs)   # 应用dropout

            # 计算加权值
            output = torch.matmul(attn_probs, v)  # (B, num_heads, L, head_dim)

        # 6. 连接多头并投影输出
        output = output.transpose(1,2).reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim) # 先转置回 (B, L, num_heads, head_dim)，再合并头
        output = self.o_proj(output)  # 最终线性投影
        output = self.residual_dropout(output)  # 应用残差连接的dropout （该Dropout在残差连接之前使用）


# class FNN
class FeedForward(nn.Module):
    def __init__(self, config:MyMindConfig):
        super().__init__()
        # 中间层大小
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # SwiGLU： 三个权重矩阵
        self.w_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w_up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False) # value矩阵
        self.w_down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False) # 输出矩阵
        # gate_proj: hidden -> intermediate (用于计算gate部分)
        # up_proj: hidden -> intermediate (用于被gate的部分)·
        # down_proj: intermediate -> hidden (用于投影回hidden维度)

        self.act_fn = ACT2FN[config.hidden_act] # Huggingface中的激活函数映射字典，这里是silu(更加方便)

    def forward(self, x:torch.Tensor):
        # 实现SwiGLU前向传播 = w_down( act( w_gate(x) ) * w_up(x) )
        gate = self.w_gate(x)  # 计算gate部分
        up = self.w_up(x)      # 计算被gate的部分
        res = self.act_fn(gate) * up
        return self.dropout(self.w_down(res))  # 最终投影并应用dropout
    

# MiniMind Block： Transformer Block实现
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id:int, config:MyMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads # 注意力头数
        self.hidden_size = config.hidden_size # 隐藏层大小
        self.head_dim = config.hidden_size // config.num_attention_heads # 每个头的维度
        self.layer_id = layer_id # 层编号   

        # GQA注意力层
        self.attention = Attention(config)

        # FFN层
        self.ffn = FeedForward(config)

        # RMSNorm层, 分别用于注意力和FFN
        self.input_rmsnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 注意力前归一化
        self.post_attention_rmsnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # FFN前归一化

    def forward(
            self,
            hidden_states:torch.Tensor, # (batch_size, seq_len, hidden_size) 中间输入
            position_embeddings:Tuple[torch.Tensor, torch.Tensor], # cos,sin
            past_key_value:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, # KV cache，二元组(past_key, past_value)
            use_cache:bool=False, # 是否返回缓存
            attention_mask:Optional[torch.Tensor]=None, # 注意力掩码
    ):
        # Norm -> Attention -> Residual add -> Norm -> FFN -> Residual add
        residual= hidden_states # 残差
        
        # 1. 注意力子层
        normed_states = self.input_rmsnorm(hidden_states) # 归一化
        
        # 按照Attnetion类的forward方法的参数要求调用  
        attn_output, present_key_value = self.attention(
            hidden_states=normed_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        hidden_states = residual + attn_output # 残差连接

        # 2. FFN子层
        normed_states = self.post_attention_rmsnorm(hidden_states) # 归一化
        ffn_output = self.ffn(normed_states)
        hidden_states = hidden_states + ffn_output # 残差连接 

        return hidden_states, present_key_value if use_cache else None    