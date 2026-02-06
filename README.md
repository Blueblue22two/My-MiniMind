# 复现MiniMind
源项目: [MiniMind Git link](https://github.com/jingyaogong/minimind)     
All dependencies in `pyproject.toml`. 


## 1. Config:
1. 安装uv:  
`pip install uv`. 

2. 安装所有依赖，通过uv自动安装`pyproject.toml`中所有依赖项。  
`uv sync`  

    进入虚拟环境  
`source .venv/bin/activate`  

3. 将需要的使用的训练数据放到`dataset`文件夹中。  
[MiniMind dataset url](https://www.modelscope.cn/datasets/gongjy/minimind_dataset)

---

## 2. 数据集下载
推荐使用modelscope下载  
`pip install modelscope`  

使用下面的命令下载指定的某个文件到指定路径:  
`modelscope download --dataset gongjy/minimind_dataset xxx_file_name --local_dir ./xxx_dir`

- Example:   

下载`pretrain_hq.jsonl`的数据集到路径`/root/autodl-tmp/My-MiniMind/dataset`中  
`modelscope download --dataset gongjy/minimind_dataset pretrain_hq.jsonl --local_dir /root/autodl-tmp/My-MiniMind/dataset`


## 3. 预训练  
调用预训练并启动swanlab记录日志   
`uv run python trainer/trainer_pretrain.py --use_wandb`    
需要注册swanlab账号，用下面指令登陆. 
`swanlab login`  