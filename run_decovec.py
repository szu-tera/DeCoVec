"""
DeCoVec 实验入口脚本（重构版）
使用模块化架构，代码更简洁、易于扩展
"""
import os
import argparse
from decovec.experiment_manager import ExperimentManager
from simple_config import get_config, set_config, SimpleConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行 DeCoVec 实验（重构版）")
    
    # 基本参数
    parser.add_argument(
        "--mode",
                        choices=["full", "zero_shot", "random_icl", "icl", "test_scale"],
                        default="full",
        help="实验模式"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="truthfulqa",
        choices=[
            "truthfulqa",
            "truthfulqa_pool_ablation",
            "math500",
            "math500_pool_ablation",
            "aqua_rat",
            "aqua_rat_pool_ablation"
        ],
        help="数据集选择（论文实验数据集）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            # Qwen2 family
            "qwen2-0.5b",
            "qwen2-1.5b",
            "qwen2-7b",        # 论文主要模型
            # Other models
            "yi-6b",
            "llama2-7b",
            "llama3-8b",
            "gemma2-9b",
        ],
        help="模型选择（默认使用 simple_config 中的 qwen2-7b）"
    )
    
    # 实验配置
    parser.add_argument("--eval_baseline", action="store_true", default=True,
                        help="是否评估基线（Zero-shot 和 ICL）")
    parser.add_argument("--n_shot", type=int, default=None,
                        help="ICL 示例数量（默认: 生成式任务和news_factor=10, 其他=15）。支持所有模式：random_icl, icl, test_mu")
    parser.add_argument(
        "--example_order",
        type=str,
        default="ordered",
        choices=["ordered", "reverse", "random"],
        help="ICL 示例排列顺序（ordered=保持相似度顺序，reverse=倒序，random=随机打乱）"
    )
    parser.add_argument(
        "--example_order_seed",
        type=int,
        default=None,
        help="示例随机排列的随机种子（仅在 example_order=random 时使用）"
    )
    

    
    # 评估配置
    parser.add_argument("--fast_mode", action="store_true",
                        help="快速模式：只评估前100个样本")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数")
    parser.add_argument(
        "--calibration_pool_size",
        type=int,
        default=None,
        help="限制示例池中的校准样本数（默认使用全部）"
    )
    parser.add_argument(
        "--calibration_pool_seed",
        type=int,
        default=None,
        help="截断示例池前的随机种子（默认不打乱）"
    )
    
    parser.add_argument(
        "--icl_methods",
        type=str,
        default="kate",
        help="SVD/μ测试使用的 ICL 示例策略（逗号分隔，支持 kate,random_icl,bm25,mapping_error,topk）"
    )
    parser.add_argument(
        "--vector_icl_method",
        type=str,
        default=None,
        help="用于构造任务向量的 ICL 示例选择策略（默认等于 icl_methods 的第一个值）。与 steer_icl_method 解耦，允许使用不同的 ICL 方法构造向量和进行推理"
    )
    parser.add_argument(
        "--baseline_icl_method",
        type=str,
        default=None,
        help="用于计算 δz 的 baseline prompt 的 ICL 方法（默认 None 表示使用 zero-shot，可选 kate,random_icl,bm25,mapping_error,topk）。与 vector_icl_method 配合使用，允许使用不同的 baseline 和 ICL 方法计算 delta_z"
    )
    
    # λ 值测试
    parser.add_argument(
        "--lambda_values",
        type=str,
        default=None,
        help="要测试的 λ 值（逗号分隔，例如: 0.5,1.0,1.5,2.0）"
    )
    
    # 评估配置
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="生成温度（用于生成式任务，默认使用 config.py 中的配置）"
    )
    
    # 输出保存配置
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="保存生成式数据集的模型输出结果到 CSV 文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/case",
        help="输出文件保存目录（默认: results/case）"
    )
    
    # Self-Consistency 配置
    parser.add_argument(
        "--self_consistency",
        action="store_true",
        help="启用 Self-Consistency 模式（使用 temperature=0.7 采样，保存 JSON 格式结果）"
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=None,
        help="运行轮次 ID（用于 Self-Consistency 模式的文件命名，如 result_1.json）"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 获取全局配置
    config = get_config()
    
    # 如果指定了温度，更新配置
    if args.temperature is not None:
        config.temperature = args.temperature
        print(f"✓ 使用温度: {args.temperature}")
    
    # 如果指定了模型，更新配置
    if args.model is not None:
        model_configs = {
            # Qwen2 family
            "qwen2-0.5b": {
                "model_name": "Qwen/Qwen2-0.5B",
                "model_path": "checkpoints/qwen/Qwen2-0.5B"
            },
            "qwen2-1.5b": {
                "model_name": "Qwen/Qwen2-1.5B",
                "model_path": "checkpoints/qwen/Qwen2-1.5B"
            },
            "qwen2-7b": {
                "model_name": "Qwen/Qwen2-7B",
                "model_path": "checkpoints/qwen/Qwen2-7B"
            },
            # Other models
            "yi-6b": {
                "model_name": "01-ai/Yi-6B",
                "model_path": "checkpoints/yi/Yi-6B"
            },
            "llama2-7b": {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "model_path": "checkpoints/llama/Llama-2-7b-hf"
            },
            "llama3-8b": {
                "model_name": "meta-llama/Meta-Llama-3-8B",
                "model_path": "checkpoints/llama/Meta-Llama-3-8B"
            },
            "gemma2-9b": {
                "model_name": "google/gemma-2-9b",
                "model_path": "checkpoints/gemma/gemma-2-9b"
            },
        }
        
        if args.model in model_configs:
            model_cfg = model_configs[args.model]
            config.model_name = model_cfg["model_name"]
            config.model_path = model_cfg["model_path"]
            print(f"✓ 使用模型: {args.model} ({model_cfg['model_name']})")
        else:
            print(f"✗ 错误：未知的模型: {args.model}")
            return
    
    # 创建实验管理器
    icl_methods_arg = args.icl_methods.split(',') if args.icl_methods else ["kate"]
    icl_methods = [m.strip() for m in icl_methods_arg if m.strip()]
    if not icl_methods:
        icl_methods = ["kate"]
    valid_icl_methods = {"kate", "random_icl", "bm25", "mapping_error", "topk"}
    invalid_icl = [m for m in icl_methods if m not in valid_icl_methods]
    if invalid_icl:
        print(f"✗ 错误：无效的 ICL 方法: {', '.join(invalid_icl)}")
        print("  可选项: kate, random_icl, bm25, mapping_error, topk")
        return
    
    # 检查是否为生成式数据集
    GENERATIVE_DATASETS = {
        "math500",
        "math500_pool_ablation",
        "aqua_rat",
        "aqua_rat_pool_ablation",
    }
    if args.save_outputs and args.dataset not in GENERATIVE_DATASETS:
        print(f"⚠ 警告：--save_outputs 参数仅对生成式数据集有效")
        print(f"  当前数据集 '{args.dataset}' 不是生成式数据集，将忽略 --save_outputs 参数")
        args.save_outputs = False
    
    # 检查 Self-Consistency 模式
    if args.self_consistency:
        if args.dataset not in GENERATIVE_DATASETS:
            print(f"⚠ 警告：--self_consistency 参数仅对生成式数据集有效")
            print(f"  当前数据集 '{args.dataset}' 不是生成式数据集，将忽略 --self_consistency 参数")
            args.self_consistency = False
        else:
            print(f"✓ 已启用 Self-Consistency 模式（temperature=0.7 采样）")
            if args.run_id is not None:
                print(f"  运行轮次 ID: {args.run_id}")
            else:
                print(f"  ⚠ 注意：未指定 --run_id，文件将不包含轮次标识")
    
    if args.save_outputs:
        print(f"✓ 已启用输出保存功能，结果将保存到: {args.output_dir}")
    
    manager = ExperimentManager(
        dataset=args.dataset,
        n_shot=args.n_shot,
        icl_method=icl_methods[0],
        save_outputs=args.save_outputs,
        output_dir=args.output_dir,
        self_consistency=args.self_consistency,
        run_id=args.run_id,
        example_order=args.example_order,
        example_order_seed=args.example_order_seed
    )

    calibration_pool_size = args.calibration_pool_size
    if calibration_pool_size is not None and calibration_pool_size <= 0:
        print("⚠ 警告：--calibration_pool_size 需为正数，将忽略该参数")
        calibration_pool_size = None
    manager.data_loader.calibration_pool_size = calibration_pool_size
    if calibration_pool_size is not None:
        print(f"✓ 限制示例池大小: {calibration_pool_size}")
    if args.calibration_pool_seed is not None:
        manager.data_loader.calibration_pool_seed = args.calibration_pool_seed
        print(f"✓ 示例池随机种子: {args.calibration_pool_seed}")
    
    # 根据模式运行实验
    if args.mode == "full":
        # 完整实验
        manager.run_full_experiment(eval_baseline=args.eval_baseline)
    
    elif args.mode == "zero_shot":
        # 仅评估 Zero-shot 基线
        manager.setup_models()
        calibration_data, test_data = manager.data_loader.load_splits()
        
        max_samples = 100 if args.fast_mode else args.max_samples
        manager.baseline_evaluator.evaluate_zero_shot(test_data, max_samples=max_samples)
    
    elif args.mode == "random_icl":
        # 仅评估随机 ICL 基线
        print(f"\n✓ 使用 ICL 示例数量: {manager.n_shot}")
        manager.setup_models()
        calibration_data, test_data = manager.data_loader.load_splits()
        
        max_samples = 100 if args.fast_mode else args.max_samples
        manager.baseline_evaluator.evaluate_random_icl(
            test_data,
            calibration_data,
            max_samples=max_samples
        )
    
    elif args.mode == "icl":
        # 仅评估 ICL 基线（支持 kate, bm25, random_icl）
        print(f"\n✓ 使用 ICL 示例数量: {manager.n_shot}")
        manager.setup_models()
        calibration_data, test_data = manager.data_loader.load_splits()
        
        max_samples = 100 if args.fast_mode else args.max_samples
        
        for icl_method in icl_methods:
            print(f"\n--- ICL 方法: {icl_method} ---")
            manager.set_icl_method(icl_method)
            
            # 为每个 ICL 方法构建对应的索引（kNN 或 BM25）
            embeddings, knn_index = manager.build_knn_index(calibration_data)
            
            # 预计算测试集邻居（使用当前 ICL 方法）
            precomputed_neighbors = manager.precompute_test_neighbors(test_data, calibration_data)
            
            metrics = manager.baseline_evaluator.evaluate_icl(
                test_data,
                calibration_data,
                precomputed_neighbors,
                max_samples=max_samples
            )
            
            print(f"\n结果 ({icl_method}):")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value}")
    
    elif args.mode == "test_scale":
        # 测试不同 λ 值（Scaling Factor）
        if args.lambda_values is None:
            print("✗ 错误：--mode test_scale 需要指定 --lambda_values 参数")
            print("示例: python run_decovec.py --mode test_scale --lambda_values 0.5,1.0,1.5,2.0 --use_cache")
            return
        
        # 解析 λ 值
        try:
            lambda_values = [float(x.strip()) for x in args.lambda_values.split(',')]
        except ValueError:
            print(f"✗ 错误：无效的 λ 值格式: {args.lambda_values}")
            print("示例: python run_decovec.py --mode test_scale --lambda_values 0.5,1.0,1.5,2.0")
            return
        
        print(f"✓ 选择的 λ 值: {lambda_values}")
        print(f"✓ 使用 ICL 示例数量: {manager.n_shot}")
        
        # 加载模型
        manager.setup_models()
        calibration_data, test_data = manager.data_loader.load_splits()
        
        max_samples = 100 if args.fast_mode else args.max_samples
        
        # 确定 vector_icl_method（用于构造任务向量）
        vector_icl_method = args.vector_icl_method
        if vector_icl_method is None:
            # 默认等于第一个 icl_method（保持向后兼容）
            vector_icl_method = icl_methods[0]
        else:
            # 验证 vector_icl_method 是否有效
            valid_icl_methods = {"kate", "random_icl", "bm25", "mapping_error", "topk"}
            if vector_icl_method not in valid_icl_methods:
                print(f"✗ 错误：无效的 vector_icl_method: {vector_icl_method}")
                print(f"  可选项: {', '.join(valid_icl_methods)}")
                return
        
        # 确定 baseline_icl_method（用于计算 δz 的 baseline prompt）
        baseline_icl_method = args.baseline_icl_method
        if baseline_icl_method is not None:
            # 验证 baseline_icl_method 是否有效
            valid_icl_methods = {"kate", "random_icl", "bm25", "mapping_error", "topk"}
            if baseline_icl_method not in valid_icl_methods:
                print(f"✗ 错误：无效的 baseline_icl_method: {baseline_icl_method}")
                print(f"  可选项: {', '.join(valid_icl_methods)}")
                return
        
        for icl_method in icl_methods:
            steer_icl_method = icl_method  # 用于推理的 ICL 方法
            
            # 如果指定了 vector_icl_method，使用它；否则使用 steer_icl_method
            actual_vector_method = vector_icl_method if args.vector_icl_method is not None else steer_icl_method
            
            print(f"\n--- ICL 方法: {steer_icl_method} ---")
            if actual_vector_method != steer_icl_method:
                print(f"  steer_icl_method: {steer_icl_method}（用于推理时的 steering）")
                print(f"  vector_icl_method: {actual_vector_method}（用于构造任务向量）")
            if baseline_icl_method is not None:
                print(f"  baseline_icl_method: {baseline_icl_method}（用于计算 δz 的 baseline prompt）")
            
            manager.set_icl_method(steer_icl_method)
            
            # 为每个 ICL 方法构建对应的索引（kNN 或 BM25）
            # 注意：这里构建的索引用于 steer_icl_method（推理时使用）
            embeddings, knn_index = manager.build_knn_index(calibration_data)
            
            # 预计算测试集邻居（使用当前 ICL 方法，即 steer_icl_method）
            precomputed_neighbors = manager.precompute_test_neighbors(test_data, calibration_data)
            
            # 计算 delta_z 时使用 actual_vector_method
            delta_z_cache = manager.compute_delta_z_with_cache(
                calibration_data,
                knn_index,
                icl_method=actual_vector_method  # 使用 vector_icl_method 构造任务向量
            )
            
            # 设置 scale_tester 的 vector_icl_method、steer_icl_method 和 baseline_icl_method
            manager.scale_tester.vector_icl_method = actual_vector_method
            manager.scale_tester.steer_icl_method = steer_icl_method
            manager.scale_tester.baseline_icl_method = baseline_icl_method
            
            results = manager.scale_tester.test_lambda_values(
                lambda_values=lambda_values,
                test_data=test_data,
                calibration_data=calibration_data,
                delta_z_cache=delta_z_cache,
                max_samples=max_samples
            )
            
            if actual_vector_method != steer_icl_method:
                print(f"\n✓ λ 值测试完成（steer_icl_method: {steer_icl_method}, vector_icl_method: {actual_vector_method}）")
            else:
                print(f"\n✓ λ 值测试完成（ICL 方法: {steer_icl_method}）")
    



if __name__ == "__main__":
    main()

