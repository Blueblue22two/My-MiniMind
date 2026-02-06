import json
import torch
from torch.utils.data import Dataset
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" # 关闭tokenizer的并行加速，避免报错

class PretrainDataset(Dataset):
    """
        处理Pretrain Dataset
            实现dataset内定的方法：
            1. _len_ 返回数据集大小
            2. __getitem__定义获取单个数据的方法 
    """
    
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
        encoding = self.tokenizer(
            str(sample['text']), 
            add_special_tokens=False, # 不添加特殊token，后续手动添加
            max_length=self.max_length - 2,  # 留出位置给特殊token(bos 和 eos)
            truncation=True # 超过最大长度则截断
        ).input_ids # 只获取input_ids部分,确保为一维列表

        # 2. 构造输入序列input_ids # shape: (max_length,)   
        # 手动添加两个特殊token
        encoding = [self.tokenizer.bos_token_id] + encoding + [self.tokenizer.eos_token_id]
        # padding到最大长度
        input_ids = encoding + [self.tokenizer.pad_token_id] * (self.max_length - len(encoding))
        # 列表转换为tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)  

        # 3. 构造标签序列
        labels = input_ids.clone()
        # 将其中 所有padding部分 的标签设置为-100，计算loss时忽略这些位置 
        # 交叉熵损失种的参数nn.CrossEntropyLoss(ignore_index=-100)，会忽略这些值为-100的位置loss计算
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, labels