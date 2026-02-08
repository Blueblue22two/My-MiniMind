import json
import torch
from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false" # 关闭tokenizer的并行加速，避免报错

def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    移除空的<think>标签
    """
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


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
        self.samples = self.load_data(self.data_path) # 将所有jsonl数据存储为列表

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


class SFTDataset(Dataset):
    """
    加载处理SFT数据
    """
    
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train') # 获取所有训练
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        将原始数据转化为模型训练需要的标准化字符串格式
        """
        messages = conversations.copy()

        # 若该条数据中 存在'functions'字段，则设置为tools
        tools = conversations[0]["functions"] if (
            conversations and                          # 条件1: 对话列表非空
            conversations[0]["role"] == "system" and   # 条件2: 第一条消息是 system 角色
            conversations[0].get("functions")          # 条件3: 该消息包含 functions 字段
        ) else None
        
        # 将对话模板转化为训练所需要的格式化字符串
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        为数据生成label
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self,idx):
        # 1. 获取数据
        sample = self.samples[idx]

        # 2. 数据构造为符合SFT的模板
        conversations = pre_processing_chat(sample['conversations']) # 添加system角色的对话
        prompt = self.create_chat_prompt(conversations) # 转化为标准字符串格式
        prompt = post_processing_chat(prompt) # 移除空标签

        # 3.tokenize
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length] # 获取tokenize后的input_ids，并截断到最大长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) # 填充
        
        # 4. 生成labels
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)