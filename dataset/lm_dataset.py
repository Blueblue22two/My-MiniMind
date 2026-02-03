import json
from torch.util.data import Dataset
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" # 关闭tokenizer的并行加速，避免报错

class PretrainDataset(Dataset):
    # 实现dataset内定的方法：
    # 1. _len_ 返回数据集大小
    # 2. __getitem__定义获取单个数据的方法 

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(self.data_path)

    # 读取json数据
    def load_data(self, data_path):
        samples = [] # 列表存储所有样本
        # 读取所有文件, 按行读取json数据
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    # len
    def __len__(self):
        return len(self.samples)
    
    # getitem
    def __getitem__(self,idx):
        sample = self.samples[idx] # 获取单个样本
        
        # 1. tokenize
        # 优化思路：每次 __getitem__ 都调用 tokenizer，速度慢。
        # 改进：在 __init__ 中预 tokenize 所有样本，存为 input_ids 列表。
        encoding = self.tokenizer(
            str(sample['text']), # 对文本进行编码.先转字符串，防止报错
            max_length=self.max_length,
            padding='max_length', # 补齐到最大长度
            truncation=True, # 超过最大长度则截断
            return_tensors='pt' # 返回pytorch的tensor
        ) # 返回形状 (batch_size=1, seq_len)


        # 2. 转化为一维tensor
        input_ids = encoding['input_ids'].squeeze(0) # 去掉batch维度

        # 生成loss mask(与input_ids形状相同的bool张量，pad部分为0，其余为1)
        # 作用：在计算损失时，忽略 padding 位置的预测，避免模型学习“预测 pad token”。
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 3. 构造输入和标签
        X = torch.tensor(input_ids[:-1], dtype=torch.long) # 构造输入序列
        Y = torch.tensor(input_ids[1:], dtype=torch.long) # 构造标签序列，右移一位

        # 确保loss_mask与Y在位置上对齐（去掉第一个位置），避免后续损失函数计算错误
        # loss值是X与Y计算误差，Y是真实标签所以每一项都直接依赖于Y而不是X，因此loss_mask应该与Y对齐
        # Example:
        # input_ids: [A, B, C, PAD, PAD]
        # X:         [A, B, C, PAD]
        # Y:         [B, C, PAD, PAD]
        # loss_mask: [1, 1, 1, 0] -> 应该与Y对齐
        # 否则在 i=2 时间步/位置上，模型预测 C 的损失会被错误地忽略，因为 loss_mask 在该位置是 0。
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
