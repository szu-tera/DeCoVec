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
            print("[OK] Loading index from cache")
            self.bm25_index = cache_data["bm25_index"]
            self.bm25_corpus = cache_data["bm25_corpus"]
            self.calibration_data = calibration_data
            return self.bm25_index
        
        # Extract text
        texts = self._extract_text_for_indexing(calibration_data)
        
        # Preprocess text (tokenization)
        print(f"Preprocessing text for {len(texts)} samples...")
        tokenized_corpus = [self._preprocess_text(text) for text in texts]
        
        print(f"[OK] Corpus preprocessing complete")
        
        # Build BM25 index
        print(f"\nBuilding BM25 index...")
        bm25_index = BM25Okapi(tokenized_corpus)
        
        print("[OK] BM25 index construction complete")
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
            print("[OK] Loading index from cache")
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
        
        print(f"[OK] Embeddings shape: {embeddings.shape}")
        
        # Build kNN index
        print(f"\nBuilding kNN index (k={self.n_shot + 1})...")
        knn_index = NearestNeighbors(
            n_neighbors=self.n_shot + 1,  # +1 because nearest neighbor is self
            metric='cosine',
            algorithm='brute'  # Brute force is faster for small datasets
        )
        knn_index.fit(embeddings)
        
        print("[OK] kNN index construction complete")
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
            calibration_data: Calibration set data (used as example pool)
            max_neighbors: Maximum neighbors to precompute (default: n_shot * 2)
        
        Returns:
            precomputed_neighbors: Dictionary mapping {sample_idx: [neighbor_indices]}
        """
        if max_neighbors is None:
            max_neighbors = self.n_shot * 2  # Reserve some buffer
        
        print("\n" + "=" * 80)
        print("Precomputing nearest neighbors for all samples")
        print("=" * 80)
        
        # Use different indexing methods based on selection mode
        if self.selection_mode == "topk":
            # TopK method: select nearest neighbors based on original dataset index order
            # No precomputation needed, select directly based on indices
            precomputed_neighbors = {}
            for i in range(len(all_data)):
                # Select samples with indices closest to i
                n_examples = min(max_neighbors, len(calibration_data))
                candidate_indices = []
                
                # Expand in both directions from i
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
                
                # If not enough, fill from edges
                if len(candidate_indices) < n_examples:
                    for idx in range(len(calibration_data)):
                        if idx not in candidate_indices:
                            candidate_indices.append(idx)
                            if len(candidate_indices) >= n_examples:
                                break
                
                # Sort by index order
                candidate_indices = sorted(candidate_indices[:n_examples])
                precomputed_neighbors[i] = candidate_indices
            
            print(f"[OK] TopK precomputation complete for {len(precomputed_neighbors)} samples")
            if len(precomputed_neighbors) > 0:
                print(f"  Reserved {len(precomputed_neighbors[0])} candidate neighbors per sample")
            print("=" * 80)
            
            # Save to instance variables
            self.precomputed_neighbors = precomputed_neighbors
            
            return precomputed_neighbors
        
        if self.selection_mode == "bm25":
            # Use BM25 index
            if self.bm25_index is None:
                raise ValueError("Must call build_bm25_index() first to build the index")
            
            # Extract query text
            query_texts = self._extract_text_for_indexing(all_data)
            
            # Preprocess query text
            print(f"Preprocessing text for {len(query_texts)} query samples...")
            tokenized_queries = [self._preprocess_text(text) for text in query_texts]
            
            # Batch search for nearest neighbors
            print(f"\nBatch searching nearest neighbors (k={max_neighbors})...")
            precomputed_neighbors = {}
            for i, query_tokens in enumerate(tokenized_queries):
                # Get BM25 scores
                scores = self.bm25_index.get_scores(query_tokens)
                # Sort by score descending and get top-k
                top_indices = np.argsort(scores)[::-1][:min(max_neighbors, len(calibration_data))]
                precomputed_neighbors[i] = top_indices.tolist()
            
            print(f"[OK] Precomputation complete for {len(precomputed_neighbors)} samples")
            if len(precomputed_neighbors) > 0:
                print(f"  Reserved {len(precomputed_neighbors[0])} candidate neighbors per sample")
            print("=" * 80)
            
            # 保存到实例变量
            self.precomputed_neighbors = precomputed_neighbors
            
            return precomputed_neighbors
        
        else:
            # Use kNN index (original logic)
            # Ensure index has been built
            if self.knn_index is None or self.embeddings is None:
                raise ValueError("Must call build_knn_index() first to build the index")
            
            # Compute embeddings for all query samples
            print(f"Computing embeddings for {len(all_data)} query samples...")
            # For news_factor, use same strategy as build_knn_index (directly use all choices)
            if self.dataset_type == "news_factor":
                queries = []
                for item in all_data:
                    choices = item.get("choices", [])
                    # Concatenate all choices
                    choices_text = " ".join(choices)
                    queries.append(choices_text)
            else:
                queries = [item["question"] for item in all_data]
            query_embeddings = self.emb_model.encode(
                queries,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            print(f"[OK] Query embeddings shape: {query_embeddings.shape}")
            
            # Batch search for nearest neighbors
            print(f"\nBatch searching nearest neighbors (k={max_neighbors})...")
            distances, indices = self.knn_index.kneighbors(
                query_embeddings,
                n_neighbors=min(max_neighbors, len(calibration_data))
            )
            
            # Save results
            precomputed_neighbors = {}
            for i in range(len(all_data)):
                # Save sorted neighbor index list (already sorted by distance from near to far)
                precomputed_neighbors[i] = indices[i].tolist()
            
            print(f"[OK] Precomputation complete for {len(precomputed_neighbors)} samples")
            print(f"  Reserved {len(precomputed_neighbors[0])} candidate neighbors per sample")
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
        Get ICL examples for a given sample (using kNN or precomputed results)
        
        Args:
            query_idx: Index of the query sample
            calibration_data: Calibration set data (if None, use saved data)
            use_precomputed: Whether to use precomputed neighbors (if available)
        
        Returns:
            examples: List of selected examples
        """
        if calibration_data is None:
            calibration_data = self.calibration_data
        
        if calibration_data is None:
            raise ValueError("No calibration data available")
        
        if self.selection_mode == "random_icl":
            examples = self._get_random_examples(query_idx, calibration_data)
            return self._apply_example_order(examples, query_idx)
        
        if self.selection_mode == "mapping_error":
            examples = self._get_mapping_error_examples(query_idx, calibration_data)
            return self._apply_example_order(examples, query_idx)
        
        if self.selection_mode == "topk":
            examples = self._get_topk_examples(query_idx, calibration_data)
            return self._apply_example_order(examples, query_idx)
        
        # If precomputed results exist, use directly (faster)
        if use_precomputed and self.precomputed_neighbors is not None:
            if query_idx not in self.precomputed_neighbors:
                raise ValueError(f"Query index {query_idx} has no precomputed neighbors")
            
            # Get first n_shot neighbors from precomputed results
            neighbor_indices = self.precomputed_neighbors[query_idx][:self.n_shot]
            examples = [calibration_data[idx] for idx in neighbor_indices]
            return self._apply_example_order(examples, query_idx)
        
        # Compute in real time based on selection mode
        if self.selection_mode == "bm25":
            # Use BM25 retrieval
            if self.bm25_index is None:
                raise ValueError("Must call build_bm25_index() first to build the index")
            
            # Extract query text (query_idx should point to index in calibration_data)
            if query_idx >= len(calibration_data):
                raise ValueError(f"Query index {query_idx} exceeds calibration set size {len(calibration_data)}")
            
            query_item = calibration_data[query_idx]
            query_texts = self._extract_text_for_indexing([query_item])
            if not query_texts or not query_texts[0]:
                raise ValueError(f"Failed to extract query text, index: {query_idx}")
            
            # Preprocess query text
            query_tokens = self._preprocess_text(query_texts[0])
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Sort by score descending and get top-n_shot
            top_indices = np.argsort(scores)[::-1][:self.n_shot + 1]  # +1 because might include self
            
            # Exclude self (if in calibration set)
            top_indices = [idx for idx in top_indices if idx != query_idx][:self.n_shot]
            
            examples = [calibration_data[idx] for idx in top_indices]
            return self._apply_example_order(examples, query_idx)
        
        else:
            # Use kNN retrieval (original logic)
            if self.knn_index is None:
                raise ValueError("Must call build_knn_index() first to build the index")
            
            # Get nearest neighbors (first is self, skip it)
            distances, indices = self.knn_index.kneighbors(
                [self.embeddings[query_idx]],
                n_neighbors=self.n_shot + 1
            )
            
            # Skip first (self)
            neighbor_indices = indices[0][1:self.n_shot + 1]
            
            # Return examples
            examples = [calibration_data[idx] for idx in neighbor_indices]
            return self._apply_example_order(examples, query_idx)

    def set_selection_mode(self, mode: str):
        """Set example selection mode"""
        valid_modes = {"kate", "random_icl", "bm25", "mapping_error", "topk"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown ICL method: {mode}, options: {', '.join(valid_modes)}")
        self.selection_mode = mode

    def _apply_example_order(self, examples: List[Dict], query_idx: Optional[int] = None) -> List[Dict]:
        """Adjust example order based on configuration"""
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
        """Randomly select ICL examples"""
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
        Randomly select ICL examples and randomize labels (Mapping Error method)
        
        Tests whether the model relies on context evidence (Context) or pre-trained priors (Priors).
        If the model primarily relies on pre-trained priors, performance won't drop significantly
        even if labels are randomized.
        
        Args:
            query_idx: Query sample index
            calibration_data: Calibration set data
            seed: Random seed
        
        Returns:
            examples: List of selected examples (labels randomized)
        """
        if seed is None:
            seed = query_idx
        rng = np.random.RandomState(seed)
        available_indices = list(range(len(calibration_data)))
        n_examples = min(self.n_shot, len(available_indices))
        if n_examples == 0:
            return []
        
        # Randomly select examples
        selected_indices = rng.choice(available_indices, size=n_examples, replace=False)
        examples = []
        
        for idx in selected_indices:
            # Deep copy example to avoid modifying original data
            example = calibration_data[idx].copy()
            
            # Randomize labels based on dataset type
            if self.dataset_type == "boolq":
                # BoolQ: answer field, options ["true", "false"]
                original_answer = example.get("answer", "true")
                choices = ["true", "false"]
                # Randomly select a different answer if possible
                wrong_choices = [c for c in choices if c != original_answer]
                if wrong_choices:
                    example["answer"] = rng.choice(wrong_choices)
                else:
                    example["answer"] = rng.choice(choices)
            
            elif self.dataset_type == "commonsense_qa":
                # CommonsenseQA: answer field, options from labels list
                labels = example.get("labels", ["A", "B", "C", "D", "E"])
                original_answer = example.get("answer", labels[0] if labels else "A")
                # Randomly select a different label
                wrong_labels = [l for l in labels if l != original_answer]
                if wrong_labels:
                    example["answer"] = rng.choice(wrong_labels)
                else:
                    example["answer"] = rng.choice(labels)
                # Also update answer_text if it exists
                if "answer_text" in example and "choices" in example:
                    answer_idx = labels.index(example["answer"])
                    if answer_idx < len(example["choices"]):
                        example["answer_text"] = example["choices"][answer_idx]
            
            elif self.dataset_type == "truthfulqa":
                # TruthfulQA: best_answer field, options from mc1_choices or mc2_choices
                mc1_choices = example.get("mc1_choices", [])
                if isinstance(mc1_choices, str):
                    try:
                        mc1_choices = eval(mc1_choices)
                    except:
                        mc1_choices = []
                
                original_answer = example.get("best_answer", "")
                # Prefer mc1_choices; if not available, use mc2_choices
                available_choices = mc1_choices if mc1_choices else example.get("mc2_choices", [])
                if isinstance(available_choices, str):
                    try:
                        available_choices = eval(available_choices)
                    except:
                        available_choices = []
                
                if available_choices:
                    # Randomly select a different answer
                    wrong_choices = [c for c in available_choices if c != original_answer]
                    if wrong_choices:
                        example["best_answer"] = rng.choice(wrong_choices)
                    else:
                        example["best_answer"] = rng.choice(available_choices)
            
            elif self.dataset_type == "news_factor":
                # NEWS-FACTOR: answer field, options from labels list
                labels = example.get("labels", ["A", "B", "C", "D"])
                original_answer = example.get("answer", labels[0] if labels else "A")
                # Randomly select a different label
                wrong_labels = [l for l in labels if l != original_answer]
                if wrong_labels:
                    example["answer"] = rng.choice(wrong_labels)
                else:
                    example["answer"] = rng.choice(labels)
                # Also update answer_text and best_answer if they exist
                if "answer_text" in example and "choices" in example:
                    answer_idx = labels.index(example["answer"])
                    if answer_idx < len(example["choices"]):
                        example["answer_text"] = example["choices"][answer_idx]
                        example["best_answer"] = example["choices"][answer_idx]
            
            elif self.dataset_type == "strategyqa":
                # StrategyQA: answer field, options ["Yes", "No"]
                original_answer = example.get("answer", "Yes")
                choices = ["Yes", "No"]
                wrong_choices = [c for c in choices if c != original_answer]
                if wrong_choices:
                    example["answer"] = rng.choice(wrong_choices)
                else:
                    example["answer"] = rng.choice(choices)
            
            # For generative datasets (gsm8k, math500, svamp, asdiv, aqua_rat),
            # label randomization is not performed for now due to complex answer format.
            # Can be extended in the future if needed.
            
            examples.append(example)
        
        return examples
    
    def _get_topk_examples(
        self,
        query_idx: int,
        calibration_data: List[Dict],
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Select nearest neighbor examples based on original dataset index order (TopK method)
        
        Use the nearest neighbors of a given test sample as corresponding context examples.
        Here, "nearest neighbors" refers to samples with indices close to each other in the original dataset.
        
        Args:
            query_idx: Query sample index (in test set)
            calibration_data: Calibration set data
            seed: Random seed (unused, kept for interface consistency)
        
        Returns:
            examples: List of selected examples (ordered by index)
        """
        n_examples = min(self.n_shot, len(calibration_data))
        if n_examples == 0:
            return []
        
        # Select neighbors based on original dataset index
        # Strategy: select n_shot samples from calibration set with indices closest to query_idx
        # If query_idx exceeds calibration set size, select from both ends
        
        calibration_size = len(calibration_data)
        
        # Compute candidate index range
        # Prioritize selecting samples near query_idx; if insufficient, fill from both ends
        candidate_indices = []
        
        # First try selecting samples near query_idx
        half_shot = n_examples // 2
        
        # Expand in both directions from query_idx
        for offset in range(n_examples):
            # First try query_idx - offset (left side)
            left_idx = query_idx - offset
            if 0 <= left_idx < calibration_size and left_idx not in candidate_indices:
                candidate_indices.append(left_idx)
                if len(candidate_indices) >= n_examples:
                    break
            
            # Then try query_idx + offset (right side)
            if offset > 0:  # Avoid duplicate adding query_idx
                right_idx = query_idx + offset
                if 0 <= right_idx < calibration_size and right_idx not in candidate_indices:
                    candidate_indices.append(right_idx)
                    if len(candidate_indices) >= n_examples:
                        break
        
        # If candidate indices insufficient, fill from both ends
        if len(candidate_indices) < n_examples:
            # Fill from start
            for idx in range(calibration_size):
                if idx not in candidate_indices:
                    candidate_indices.append(idx)
                    if len(candidate_indices) >= n_examples:
                        break
            
            # If still insufficient, fill from end
            if len(candidate_indices) < n_examples:
                for idx in range(calibration_size - 1, -1, -1):
                    if idx not in candidate_indices:
                        candidate_indices.append(idx)
                        if len(candidate_indices) >= n_examples:
                            break
        
        # Sort by index order to ensure selecting "nearest neighbors"
        candidate_indices = sorted(candidate_indices[:n_examples])
        
        # Return examples
        examples = [calibration_data[idx] for idx in candidate_indices]
        return examples
    
    def get_icl_examples_for_query(
        self,
        query_text: str,
        calibration_data: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Get ICL examples for a given query text
        
        Args:
            query_text: Query text
            calibration_data: Calibration set data (if None, use saved data)
        
        Returns:
            examples: List of selected examples
        """
        if self.knn_index is None:
            raise ValueError("Must call build_knn_index() first to build the index")
        
        if calibration_data is None:
            calibration_data = self.calibration_data
        
        if calibration_data is None:
            raise ValueError("No calibration data available")
        
        # Compute embedding for query
        query_emb = self.emb_model.encode([query_text])[0]
        
        # Get nearest neighbors
        distances, indices = self.knn_index.kneighbors(
            [query_emb],
            n_neighbors=self.n_shot
        )
        
        # Return examples
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
        Construct ICL prompt (using prompt_loader)
        
        Automatically build correct format ICL prompt based on dataset type.
        
        Args:
            examples: List of examples selected by KATE
            query_item: Query sample (complete Dict)
            include_query_answer: Whether to include query answer (for calibration)
            query_answer: Query answer
        
        Returns:
            prompt: Complete prompt
        """
        # Use unified construction function from prompt_loader
        prompt = construct_icl_prompt_util(
            examples,
            query_item,
            self.prompt_config,
            self.dataset_type,
            include_query_answer,
            query_answer
        )
        
        # Check length; truncate KATE examples if too long
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > self.max_demo_tokens + 100:
                # Reduce number of KATE examples
                if len(examples) > 1:
                    return self.construct_icl_prompt(
                        examples[:-1],  # Remove last KATE example
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
        Construct zero-shot prompt (using prompt_loader)
        
        Automatically build correct format zero-shot prompt based on dataset type.
        
        Args:
            query_item: Query sample (complete Dict)
            include_answer: Whether to include answer
            answer: Answer text
            dataset_type: Dataset type
        
        Returns:
            prompt: Zero-shot prompt
        """
        # Load corresponding prompt configuration
        prompt_config = load_prompt_config(dataset_type)
        
        # If answer should be included, use ICL demo format
        if include_answer and answer:
            query_with_answer = query_item.copy()
            # Set answer field based on dataset type
            if dataset_type == "truthfulqa":
                query_with_answer["best_answer"] = answer
            else:
                query_with_answer["answer"] = answer
            
            # Format as ICL demo
            demo = format_icl_demo(query_with_answer, prompt_config, dataset_type)
            
            # If config has instruction, need to add at front (keep consistency with zero-shot format)
            if "instruction" in prompt_config:
                instruction = prompt_config["instruction"]
                return instruction + " " + demo
            else:
                return demo
        else:
            # Otherwise use zero-shot format
            from prompt_loader import format_zero_shot_prompt  # type: ignore
            return format_zero_shot_prompt(query_item, prompt_config, dataset_type)

