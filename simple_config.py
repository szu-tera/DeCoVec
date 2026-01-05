"""
简化配置模块 - 仅保留论文相关参数
用于 ExperimentManager 等内部模块
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimpleConfig:
    """DeCoVec 实验配置（仅保留论文需要的参数）"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen2-7B"
    model_path: str = "checkpoints/qwen/Qwen2-7B"
    device: str = "cuda:0"
    torch_dtype: str = "float16"  # "float16" or "bfloat16"
    
    # Embedding 模型（用于 KATE 示例选择）
    emb_model_path: str = "checkpoints/emb_models/all-MiniLM-L6-v2"
    emb_model_name: str = "all-MiniLM-L6-v2"  # fallback
    
    # 评估配置
    batch_size: int = 8
    temperature: float = 0.0  # 生成式任务的温度（0=greedy）
    max_samples: Optional[int] = None  # None=评估全部样本
    results_dir: str = "results"
    
    # 数据集配置
    data_dir: str = "data"
    seed: int = 42  # 默认随机种子
    use_full_test_set: bool = True  # 使用完整测试集
    
    # 数据集特定随机种子（论文实验配置）
    dataset_seeds: dict = None
    
    # 实验名称
    experiment_name: str = "decovec_experiment"
    
    def __post_init__(self):
        """初始化数据集种子字典"""
        if self.dataset_seeds is None:
            self.dataset_seeds = {
                "truthful_qa": 44,
                "truthfulqa": 44,
                "boolq": 42,
                "commonsense_qa": 42,
                "gsm8k": 42,
                "math500": 42,
                "aqua_rat": 42,
            }
        
        # 创建必要的目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_seed(self, dataset_name: str) -> int:
        """获取数据集对应的随机种子"""
        return self.dataset_seeds.get(dataset_name, self.seed)


# 全局配置实例
_config = SimpleConfig()


def get_config():
    """获取全局配置"""
    return _config


def set_config(new_config: SimpleConfig):
    """设置全局配置"""
    global _config
    _config = new_config
