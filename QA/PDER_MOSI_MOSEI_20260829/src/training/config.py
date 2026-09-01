"""
训练配置模块
"""
import os
import json
import torch
import numpy as np
import random
from typing import Dict, Any, Optional

def set_seed(seed: int) -> None:
    """
    设置随机种子，确保实验可重复
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
        
def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    获取设备（CPU/GPU）
    
    Args:
        gpu_id: GPU ID，默认为None（使用第一个可用的GPU）
        
    Returns:
        device: PyTorch设备
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    return device

def create_experiment_dir(config: Dict[str, Any]) -> str:
    """
    创建实验目录
    
    Args:
        config: 配置字典
        
    Returns:
        exp_dir: 实验目录路径
    """
    # 基础目录
    checkpoint_dir = config['paths']['checkpoint_dir']
    
    # 创建实验目录
    model_name = config['model_name']
    exp_name = f"{model_name}_{config['random_seed']}"
    exp_dir = os.path.join(checkpoint_dir, exp_name)
    
    # 创建目录
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(exp_dir, 'config.json'))
    
    return exp_dir 
