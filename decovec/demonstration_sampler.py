"""
ICL Demonstration Sampler (KATE Method)

Responsible for:
1. Building kNN index (based on sentence embeddings)
2. Selecting most similar examples for query samples
3. Constructing ICL prompts
"""
import numpy as np
import sys
import os
import re
from typing import List, Dict, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Add data directory to path for importing prompt_loader
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
if data_dir not in sys.path:
    sys.path.insert(0, data_dir)
from prompt_loader import load_prompt_config, format_icl_demo, format_icl_query, construct_icl_prompt as construct_icl_prompt_util  # type: ignore


class DemonstrationSampler:
    """ICL Demonstration Sampler (kNN-based KATE method)"""
    
    def __init__(
        self,
        emb_model: SentenceTransformer,
        n_shot: int = 15,
        max_demo_tokens: int = 1024,
        tokenizer = None,
        dataset_type: str = "truthfulqa",
        selection_mode: str = "kate",
        example_order: str = "ordered",
        example_order_seed: Optional[int] = None
    ):
        """
        Initialize the demonstration sampler
        
        Args:
            emb_model: Sentence embedding model
            n_shot: Number of examples to select per sample (recommended: gsm8k=10, others=15)
            max_demo_tokens: Maximum tokens for demonstration portion (recommended: gsm8k=2048, others=1024)
            tokenizer: Used for checking token length
            dataset_type: Dataset type ("truthfulqa", "boolq", "gsm8k", "commonsense_qa")
        """
        self.emb_model = emb_model
        self.n_shot = n_shot
        self.max_demo_tokens = max_demo_tokens
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.knn_index = None
        self.embeddings = None
        self.calibration_data = None
        self.precomputed_neighbors = None  # Precomputed nearest neighbors {sample_idx: [neighbor_indices]}
        self.selection_mode = selection_mode
        self.example_order = example_order  # ordered | reverse | random
        self.example_order_seed = example_order_seed
        
        # BM25 related attributes
        self.bm25_index = None
        self.bm25_corpus = None  # Tokenized corpus
        
        # Load prompt configuration for dataset
        self.prompt_config = load_prompt_config(dataset_type)
    
    @staticmethod
    def _preprocess_text(text: str) -> List[str]:
        """
        Text preprocessing: tokenization, lowercasing, punctuation removal
        
        Args:
            text: Original text
            
        Returns:
            tokens: List of tokenized words
        """
        # Convert to lowercase
        text = text.lower()
        # Tokenize using regex (keep letters and digits)
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _extract_text_for_indexing(self, data: List[Dict]) -> List[str]:
        """
        Extract text for indexing from data
        
        Args:
            data: Data list
            
        Returns:
            texts: List of texts
        """
        texts = []
        for item in data:
            if self.dataset_type == "news_factor":
                # For news_factor, use combined text of all choices
                choices = item.get("choices", [])
                choices_text = " ".join(choices)
                texts.append(choices_text)
            else:
                # Other datasets use question field
                texts.append(item.get("question", ""))
        return texts
    
    def build_bm25_index(
        self,
        calibration_data: List[Dict],
        use_cache: bool = False,
        cache_data: Optional[Dict] = None
    ) -> Any:
        """
        Build BM25 index on calibration set
        
        Args:
            calibration_data: Calibration set data
            use_cache: Whether to use cache
            cache_data: Cached data (containing bm25_index and bm25_corpus)
        
        Returns:
            bm25_index: BM25Okapi index object
        """
        if BM25Okapi is None:
            raise ImportError("rank-bm25 library not installed, please run: pip install rank-bm25")
        
        print("\n" + "=" * 80)
        print("Building BM25 Index")
        print("=" * 80)
        
        # If cache data provided, use directly
        if use_cache and cache_data is not None:
            print("✓ Loading index from cache")
            self.bm25_index = cache_data["bm25_index"]
            self.bm25_corpus = cache_data["bm25_corpus"]
            self.calibration_data = calibration_data
            return self.bm25_index
        
        # Extract text
        texts = self._extract_text_for_indexing(calibration_data)
        
        # Preprocess text (tokenization)
        print(f"Preprocessing text for {len(texts)} samples...")
        tokenized_corpus = [self._preprocess_text(text) for text in texts]
        
        print(f"✓ Corpus preprocessing complete")
        
        # Build BM25 index
        print(f"\nBuilding BM25 index...")
        bm25_index = BM25Okapi(tokenized_corpus)
        
        print("✓ BM25 index construction complete")
        print("=" * 80)
        
        # Save to instance variables
        self.bm25_index = bm25_index
        self.bm25_corpus = tokenized_corpus
        self.calibration_data = calibration_data
        
        return bm25_index
    
    def build_knn_index(
        self,
        calibration_data: List[Dict],
        use_cache: bool = False,
        cache_data: Optional[Dict] = None
    ) -> Tuple[np.ndarray, NearestNeighbors]:
        """
        Build kNN index on calibration set
        
        Args:
            calibration_data: Calibration set data
            use_cache: Whether to use cache
            cache_data: Cached data (containing embeddings and knn_index)
        
        Returns:
            embeddings: Embeddings for all samples [N, D]
            knn_index: sklearn NearestNeighbors index
        """
        print("\n" + "=" * 80)
        print("Building kNN Index (Embedding-based)")
        print("=" * 80)
        
        # If cache data provided, use directly
        if use_cache and cache_data is not None:
            print("✓ Loading index from cache")
            self.embeddings = cache_data["embeddings"]
            self.knn_index = cache_data["knn_index"]
            self.calibration_data = calibration_data
            return self.embeddings, self.knn_index
        
        # Extract question text (for similarity computation)
        # For news_factor, directly use all choices embedding for similarity computation
        if self.dataset_type == "news_factor":
            # Use combined text of all choices for similarity computation
            questions = []
            for item in calibration_data:
                choices = item.get("choices", [])
                # Combine all choices
                choices_text = " ".join(choices)
                questions.append(choices_text)
        else:
            questions = [item["question"] for item in calibration_data]
        
        print(f"Computing embeddings for {len(questions)} samples...")
        embeddings = self.emb_model.encode(
            questions,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ Embeddings shape: {embeddings.shape}")
        
        # Build kNN index
        print(f"\nBuilding kNN index (k={self.n_shot + 1})...")
        knn_index = NearestNeighbors(
            n_neighbors=self.n_shot + 1,  # +1 because nearest neighbor is self
            metric='cosine',
            algorithm='brute'  # Brute force is faster for small datasets
        )
        knn_index.fit(embeddings)
        
        print("✓ kNN index construction complete")
        print("=" * 80)
        
        # Save to instance variables
        self.embeddings = embeddings
        self.knn_index = knn_index
        self.calibration_data = calibration_data
        
        return embeddings, knn_index
    
    def precompute_all_neighbors(
        self,
        all_data: List[Dict],
        calibration_data: List[Dict],
        max_neighbors: int = None
    ) -> Dict[int, List[int]]:
        """
        Precompute nearest neighbors for all samples (including test set)
        
        This way evaluation doesn't need to repeatedly compute embeddings and search nearest neighbors,
        just look up the pre-sorted neighbor list.
        
        Args:
            all_data: All data to compute nearest neighbors for (usually test set)
            calibration_data: 校准集数据（作为示例库）
            max_neighbors: 预计算的最大邻居数（默认为 n_shot * 2）
        
        Returns:
            precomputed_neighbors: {sample_idx: [neighbor_indices]} 字典
        """
        if max_neighbors is None:
            max_neighbors = self.n_shot * 2  # 预留一些备用
        
        print("\n" + "=" * 80)
        print("预计算所有样本的最近邻")
        print("=" * 80)
        
        # 根据选择模式使用不同的索引方法
        if self.selection_mode == "topk":
            # TopK 方法：基于原始数据集索引顺序选择最近邻
            # 不需要预计算，直接基于索引选择
            precomputed_neighbors = {}
            for i in range(len(all_data)):
                # 选择索引最接近 i 的样本
                n_examples = min(max_neighbors, len(calibration_data))
                candidate_indices = []
                
                # 从 i 开始，向两边扩展
                for offset in range(n_examples):
                    left_idx = i - offset
                    if 0 <= left_idx < len(calibration_data) and left_idx not in candidate_indices:
                        candidate_indices.append(left_idx)
                        if len(candidate_indices) >= n_examples:
                            break
                    
                    if offset > 0:
                        right_idx = i + offset
                        if 0 <= right_idx < len(calibration_data) and right_idx not in candidate_indices:
                            candidate_indices.append(right_idx)
                            if len(candidate_indices) >= n_examples:
                                break
                
                # 如果不够，从两端补充
                if len(candidate_indices) < n_examples:
                    for idx in range(len(calibration_data)):
                        if idx not in candidate_indices:
                            candidate_indices.append(idx)
                            if len(candidate_indices) >= n_examples:
                                break
                
                # 按索引顺序排序
                candidate_indices = sorted(candidate_indices[:n_examples])
                precomputed_neighbors[i] = candidate_indices
            
            print(f"✓ TopK 预计算完成，共 {len(precomputed_neighbors)} 个样本")
            if len(precomputed_neighbors) > 0:
                print(f"  每个样本预留 {len(precomputed_neighbors[0])} 个候选邻居")
            print("=" * 80)
            
            # 保存到实例变量
            self.precomputed_neighbors = precomputed_neighbors
            
            return precomputed_neighbors
        
        if self.selection_mode == "bm25":
            # 使用 BM25 索引
            if self.bm25_index is None:
                raise ValueError("必须先调用 build_bm25_index() 构建索引")
            
            # 提取查询文本
            query_texts = self._extract_text_for_indexing(all_data)
            
            # 预处理查询文本
            print(f"预处理 {len(query_texts)} 个查询样本的文本...")
            tokenized_queries = [self._preprocess_text(text) for text in query_texts]
            
            # 批量搜索最近邻
            print(f"\n批量搜索最近邻（k={max_neighbors}）...")
            precomputed_neighbors = {}
            for i, query_tokens in enumerate(tokenized_queries):
                # 获取 BM25 分数
                scores = self.bm25_index.get_scores(query_tokens)
                # 按分数降序排序，获取 top-k
                top_indices = np.argsort(scores)[::-1][:min(max_neighbors, len(calibration_data))]
                precomputed_neighbors[i] = top_indices.tolist()
            
            print(f"✓ 预计算完成，共 {len(precomputed_neighbors)} 个样本")
            if len(precomputed_neighbors) > 0:
                print(f"  每个样本预留 {len(precomputed_neighbors[0])} 个候选邻居")
            print("=" * 80)
            
            # 保存到实例变量
            self.precomputed_neighbors = precomputed_neighbors
            
            return precomputed_neighbors
        
        else:
            # 使用 kNN 索引（原有逻辑）
            # 确保已经构建了索引
            if self.knn_index is None or self.embeddings is None:
                raise ValueError("必须先调用 build_knn_index() 构建索引")
            
            # 计算所有查询样本的 embeddings
            print(f"计算 {len(all_data)} 个查询样本的 embeddings...")
            # 对于 news_factor，使用与 build_knn_index 相同的策略（直接使用所有选项）
            if self.dataset_type == "news_factor":
                queries = []
                for item in all_data:
                    choices = item.get("choices", [])
                    # 将所有选项拼接起来
                    choices_text = " ".join(choices)
                    queries.append(choices_text)
            else:
                queries = [item["question"] for item in all_data]
            query_embeddings = self.emb_model.encode(
                queries,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            print(f"✓ 查询 embeddings shape: {query_embeddings.shape}")
            
            # 批量搜索最近邻
            print(f"\n批量搜索最近邻（k={max_neighbors}）...")
            distances, indices = self.knn_index.kneighbors(
                query_embeddings,
                n_neighbors=min(max_neighbors, len(calibration_data))
            )
            
            # 保存结果
            precomputed_neighbors = {}
            for i in range(len(all_data)):
                # 保存排好序的邻居索引列表（已经按距离从近到远排序）
                precomputed_neighbors[i] = indices[i].tolist()
            
            print(f"✓ 预计算完成，共 {len(precomputed_neighbors)} 个样本")
            print(f"  每个样本预留 {len(precomputed_neighbors[0])} 个候选邻居")
            print("=" * 80)
            
            # 保存到实例变量
            self.precomputed_neighbors = precomputed_neighbors
            
            return precomputed_neighbors
    
    def get_icl_examples(
        self,
        query_idx: int,
        calibration_data: Optional[List[Dict]] = None,
        use_precomputed: bool = True
    ) -> List[Dict]:
        """
        为给定样本获取 ICL 示例（使用 kNN 或预计算结果）
        
        Args:
            query_idx: 查询样本的索引
            calibration_data: 校准集数据（如果为 None 则使用已保存的）
            use_precomputed: 是否使用预计算的邻居（如果可用）
        
        Returns:
            examples: 选中的示例列表
        """
        if calibration_data is None:
            calibration_data = self.calibration_data
        
        if calibration_data is None:
            raise ValueError("没有可用的校准数据")
        
        if self.selection_mode == "random_icl":
            examples = self._get_random_examples(query_idx, calibration_data)
            return self._apply_example_order(examples, query_idx)
        
        if self.selection_mode == "mapping_error":
            examples = self._get_mapping_error_examples(query_idx, calibration_data)
            return self._apply_example_order(examples, query_idx)
        
        if self.selection_mode == "topk":
            examples = self._get_topk_examples(query_idx, calibration_data)
            return self._apply_example_order(examples, query_idx)
        
        # 如果有预计算的结果，直接使用（更快）
        if use_precomputed and self.precomputed_neighbors is not None:
            if query_idx not in self.precomputed_neighbors:
                raise ValueError(f"查询索引 {query_idx} 没有预计算的邻居")
            
            # 直接从预计算结果中获取前 n_shot 个邻居
            neighbor_indices = self.precomputed_neighbors[query_idx][:self.n_shot]
            examples = [calibration_data[idx] for idx in neighbor_indices]
            return self._apply_example_order(examples, query_idx)
        
        # 根据选择模式进行实时计算
        if self.selection_mode == "bm25":
            # 使用 BM25 检索
            if self.bm25_index is None:
                raise ValueError("必须先调用 build_bm25_index() 构建索引")
            
            # 提取查询文本（query_idx 应该指向 calibration_data 中的索引）
            if query_idx >= len(calibration_data):
                raise ValueError(f"查询索引 {query_idx} 超出校准集范围 {len(calibration_data)}")
            
            query_item = calibration_data[query_idx]
            query_texts = self._extract_text_for_indexing([query_item])
            if not query_texts or not query_texts[0]:
                raise ValueError(f"无法提取查询文本，索引: {query_idx}")
            
            # 预处理查询文本
            query_tokens = self._preprocess_text(query_texts[0])
            
            # 获取 BM25 分数
            scores = self.bm25_index.get_scores(query_tokens)
            
            # 按分数降序排序，获取 top-n_shot
            top_indices = np.argsort(scores)[::-1][:self.n_shot + 1]  # +1 因为可能包含自己
            
            # 排除自己（如果在校准集中）
            top_indices = [idx for idx in top_indices if idx != query_idx][:self.n_shot]
            
            examples = [calibration_data[idx] for idx in top_indices]
            return self._apply_example_order(examples, query_idx)
        
        else:
            # 使用 kNN 检索（原有逻辑）
            if self.knn_index is None:
                raise ValueError("必须先调用 build_knn_index() 构建索引")
            
            # 获取最近邻（第一个是自己，跳过）
            distances, indices = self.knn_index.kneighbors(
                [self.embeddings[query_idx]],
                n_neighbors=self.n_shot + 1
            )
            
            # 跳过第一个（自己）
            neighbor_indices = indices[0][1:self.n_shot + 1]
            
            # 返回示例
            examples = [calibration_data[idx] for idx in neighbor_indices]
            return self._apply_example_order(examples, query_idx)

    def set_selection_mode(self, mode: str):
        """设置示例选择模式"""
        valid_modes = {"kate", "random_icl", "bm25", "mapping_error", "topk"}
        if mode not in valid_modes:
            raise ValueError(f"未知的 ICL 方法: {mode}，可选值: {', '.join(valid_modes)}")
        self.selection_mode = mode

    def _apply_example_order(self, examples: List[Dict], query_idx: Optional[int] = None) -> List[Dict]:
        """根据配置调整示例顺序"""
        if not examples:
            return examples
        if self.example_order == "ordered":
            return examples
        if self.example_order == "reverse":
            return list(reversed(examples))
        if self.example_order == "random":
            seed_val = self.example_order_seed
            if seed_val is None and query_idx is not None:
                seed_val = query_idx
            rng = np.random.RandomState(seed_val) if seed_val is not None else np.random.RandomState()
            shuffled = examples.copy()
            rng.shuffle(shuffled)
            return shuffled
        return examples

    def _get_random_examples(
        self,
        query_idx: int,
        calibration_data: List[Dict],
        seed: Optional[int] = None
    ) -> List[Dict]:
        """随机选择 ICL 示例"""
        if seed is None:
            seed = query_idx
        rng = np.random.RandomState(seed)
        available_indices = list(range(len(calibration_data)))
        n_examples = min(self.n_shot, len(available_indices))
        if n_examples == 0:
            return []
        selected_indices = rng.choice(available_indices, size=n_examples, replace=False)
        return [calibration_data[i] for i in selected_indices]
    
    def _get_mapping_error_examples(
        self,
        query_idx: int,
        calibration_data: List[Dict],
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        随机选择 ICL 示例并随机化 label（Mapping Error 方法）
        
        用于测试模型是否依赖上下文证据（Context）还是预训练先验（Priors）。
        如果模型主要依赖预训练先验，即使 label 被随机化，性能也不会显著下降。
        
        Args:
            query_idx: 查询样本索引
            calibration_data: 校准集数据
            seed: 随机种子
        
        Returns:
            examples: 选中的示例列表（label 已被随机化）
        """
        if seed is None:
            seed = query_idx
        rng = np.random.RandomState(seed)
        available_indices = list(range(len(calibration_data)))
        n_examples = min(self.n_shot, len(available_indices))
        if n_examples == 0:
            return []
        
        # 随机选择示例
        selected_indices = rng.choice(available_indices, size=n_examples, replace=False)
        examples = []
        
        for idx in selected_indices:
            # 深拷贝示例，避免修改原始数据
            example = calibration_data[idx].copy()
            
            # 根据数据集类型随机化 label
            if self.dataset_type == "boolq":
                # BoolQ: answer 字段，可选值 ["true", "false"]
                original_answer = example.get("answer", "true")
                choices = ["true", "false"]
                # 随机选择一个不同的答案（如果可能）
                wrong_choices = [c for c in choices if c != original_answer]
                if wrong_choices:
                    example["answer"] = rng.choice(wrong_choices)
                else:
                    example["answer"] = rng.choice(choices)
            
            elif self.dataset_type == "commonsense_qa":
                # CommonsenseQA: answer 字段，可选值来自 labels 列表
                labels = example.get("labels", ["A", "B", "C", "D", "E"])
                original_answer = example.get("answer", labels[0] if labels else "A")
                # 随机选择一个不同的标签
                wrong_labels = [l for l in labels if l != original_answer]
                if wrong_labels:
                    example["answer"] = rng.choice(wrong_labels)
                else:
                    example["answer"] = rng.choice(labels)
                # 同时更新 answer_text（如果存在）
                if "answer_text" in example and "choices" in example:
                    answer_idx = labels.index(example["answer"])
                    if answer_idx < len(example["choices"]):
                        example["answer_text"] = example["choices"][answer_idx]
            
            elif self.dataset_type == "truthfulqa":
                # TruthfulQA: best_answer 字段，可选值来自 mc1_choices 或 mc2_choices
                mc1_choices = example.get("mc1_choices", [])
                if isinstance(mc1_choices, str):
                    try:
                        mc1_choices = eval(mc1_choices)
                    except:
                        mc1_choices = []
                
                original_answer = example.get("best_answer", "")
                # 优先使用 mc1_choices，如果没有则使用 mc2_choices
                available_choices = mc1_choices if mc1_choices else example.get("mc2_choices", [])
                if isinstance(available_choices, str):
                    try:
                        available_choices = eval(available_choices)
                    except:
                        available_choices = []
                
                if available_choices:
                    # 随机选择一个不同的答案
                    wrong_choices = [c for c in available_choices if c != original_answer]
                    if wrong_choices:
                        example["best_answer"] = rng.choice(wrong_choices)
                    else:
                        example["best_answer"] = rng.choice(available_choices)
            
            elif self.dataset_type == "news_factor":
                # NEWS-FACTOR: answer 字段，可选值来自 labels 列表
                labels = example.get("labels", ["A", "B", "C", "D"])
                original_answer = example.get("answer", labels[0] if labels else "A")
                # 随机选择一个不同的标签
                wrong_labels = [l for l in labels if l != original_answer]
                if wrong_labels:
                    example["answer"] = rng.choice(wrong_labels)
                else:
                    example["answer"] = rng.choice(labels)
                # 同时更新 answer_text 和 best_answer（如果存在）
                if "answer_text" in example and "choices" in example:
                    answer_idx = labels.index(example["answer"])
                    if answer_idx < len(example["choices"]):
                        example["answer_text"] = example["choices"][answer_idx]
                        example["best_answer"] = example["choices"][answer_idx]
            
            elif self.dataset_type == "strategyqa":
                # StrategyQA: answer 字段，可选值 ["Yes", "No"]
                original_answer = example.get("answer", "Yes")
                choices = ["Yes", "No"]
                wrong_choices = [c for c in choices if c != original_answer]
                if wrong_choices:
                    example["answer"] = rng.choice(wrong_choices)
                else:
                    example["answer"] = rng.choice(choices)
            
            # 对于生成式数据集（gsm8k, math500, svamp, asdiv, aqua_rat），
            # 由于答案格式复杂，暂时不进行随机化（保持原样）
            # 如果需要，可以后续扩展
            
            examples.append(example)
        
        return examples
    
    def _get_topk_examples(
        self,
        query_idx: int,
        calibration_data: List[Dict],
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        基于原始数据集索引顺序选择最近邻示例（TopK 方法）
        
        将给定测试样本的最近邻作为相应的上下文示例。
        这里的"最近邻"指的是在原始数据集中索引位置靠近的样本。
        
        Args:
            query_idx: 查询样本的索引（在测试集中的索引）
            calibration_data: 校准集数据
            seed: 随机种子（未使用，保持接口一致性）
        
        Returns:
            examples: 选中的示例列表（按索引顺序选择）
        """
        n_examples = min(self.n_shot, len(calibration_data))
        if n_examples == 0:
            return []
        
        # 基于原始数据集索引选择最近邻
        # 策略：选择校准集中索引最接近 query_idx 的 n_shot 个样本
        # 如果 query_idx 超出校准集范围，从两端选择
        
        calibration_size = len(calibration_data)
        
        # 计算候选索引范围
        # 优先选择 query_idx 附近的样本，如果不够则从两端补充
        candidate_indices = []
        
        # 首先尝试选择 query_idx 附近的样本
        half_shot = n_examples // 2
        
        # 从 query_idx 开始，向两边扩展
        for offset in range(n_examples):
            # 先尝试 query_idx - offset（左侧）
            left_idx = query_idx - offset
            if 0 <= left_idx < calibration_size and left_idx not in candidate_indices:
                candidate_indices.append(left_idx)
                if len(candidate_indices) >= n_examples:
                    break
            
            # 再尝试 query_idx + offset（右侧）
            if offset > 0:  # 避免重复添加 query_idx
                right_idx = query_idx + offset
                if 0 <= right_idx < calibration_size and right_idx not in candidate_indices:
                    candidate_indices.append(right_idx)
                    if len(candidate_indices) >= n_examples:
                        break
        
        # 如果候选索引不够，从两端补充
        if len(candidate_indices) < n_examples:
            # 从开头补充
            for idx in range(calibration_size):
                if idx not in candidate_indices:
                    candidate_indices.append(idx)
                    if len(candidate_indices) >= n_examples:
                        break
            
            # 如果还不够，从末尾补充
            if len(candidate_indices) < n_examples:
                for idx in range(calibration_size - 1, -1, -1):
                    if idx not in candidate_indices:
                        candidate_indices.append(idx)
                        if len(candidate_indices) >= n_examples:
                            break
        
        # 按索引顺序排序，确保选择的是"最近邻"
        candidate_indices = sorted(candidate_indices[:n_examples])
        
        # 返回示例
        examples = [calibration_data[idx] for idx in candidate_indices]
        return examples
    
    def get_icl_examples_for_query(
        self,
        query_text: str,
        calibration_data: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        为给定查询文本获取 ICL 示例
        
        Args:
            query_text: 查询文本
            calibration_data: 校准集数据（如果为 None 则使用已保存的）
        
        Returns:
            examples: 选中的示例列表
        """
        if self.knn_index is None:
            raise ValueError("必须先调用 build_knn_index() 构建索引")
        
        if calibration_data is None:
            calibration_data = self.calibration_data
        
        if calibration_data is None:
            raise ValueError("没有可用的校准数据")
        
        # 计算查询的 embedding
        query_emb = self.emb_model.encode([query_text])[0]
        
        # 获取最近邻
        distances, indices = self.knn_index.kneighbors(
            [query_emb],
            n_neighbors=self.n_shot
        )
        
        # 返回示例
        examples = [calibration_data[idx] for idx in indices[0]]
        return self._apply_example_order(examples)
    
    def construct_icl_prompt(
        self,
        examples: List[Dict],
        query_item: Dict,
        include_query_answer: bool = False,
        query_answer: str = None
    ) -> str:
        """
        构造 ICL 提示词（使用 prompt_loader）
        
        根据数据集类型自动构建正确格式的 ICL 提示。
        
        Args:
            examples: KATE 选择的示例列表
            query_item: 查询样本（完整 Dict）
            include_query_answer: 是否包含查询答案（用于校准）
            query_answer: 查询答案
        
        Returns:
            prompt: 完整提示词
        """
        # 使用 prompt_loader 的统一构建函数
        prompt = construct_icl_prompt_util(
            examples,
            query_item,
            self.prompt_config,
            self.dataset_type,
            include_query_answer,
            query_answer
        )
        
        # 检查长度，如果太长则截断 KATE 示例
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > self.max_demo_tokens + 100:
                # 减少 KATE 示例数量
                if len(examples) > 1:
                    return self.construct_icl_prompt(
                        examples[:-1],  # 移除最后一个 KATE 示例
                        query_item,
                        include_query_answer,
                        query_answer
                    )
        
        return prompt
    
    @staticmethod
    def construct_zero_shot_prompt(
        query_item: Dict,
        include_answer: bool = False,
        answer: str = None,
        dataset_type: str = "truthfulqa"
    ) -> str:
        """
        构造零样本提示词（使用 prompt_loader）
        
        根据数据集类型自动构建正确格式的零样本提示。
        
        Args:
            query_item: 查询样本（完整 Dict）
            include_answer: 是否包含答案
            answer: 答案文本
            dataset_type: 数据集类型
        
        Returns:
            prompt: 零样本提示词
        """
        # 加载对应的 prompt 配置
        prompt_config = load_prompt_config(dataset_type)
        
        # 如果需要包含答案，使用 ICL demo 格式
        if include_answer and answer:
            query_with_answer = query_item.copy()
            # 根据数据集类型设置答案字段
            if dataset_type == "truthfulqa":
                query_with_answer["best_answer"] = answer
            else:
                query_with_answer["answer"] = answer
            
            # 格式化 ICL demo
            demo = format_icl_demo(query_with_answer, prompt_config, dataset_type)
            
            # 如果配置中有 instruction，需要添加到前面（保持与 zero-shot 格式一致）
            if "instruction" in prompt_config:
                instruction = prompt_config["instruction"]
                return instruction + " " + demo
            else:
                return demo
        else:
            # 否则使用 zero-shot 格式
            from prompt_loader import format_zero_shot_prompt  # type: ignore
            return format_zero_shot_prompt(query_item, prompt_config, dataset_type)

