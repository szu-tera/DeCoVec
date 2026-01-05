"""
DeCoVec Core Computation Module
Contains core logic for delta-z computation, calibration, etc.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from decovec.decovec_core import TaskVectorBuilder
from decovec.demonstration_sampler import DemonstrationSampler
from evaluate.evaluate_utils import tokenize_prompt_and_continuation


class TaskVectorComputer:
    """DeCoVec Core Computer"""
    
    def __init__(
        self,
        model,
        tokenizer,
        demonstration_sampler: DemonstrationSampler,
        device: str,
        dataset: str = "truthfulqa"
    ):
        """
        Initialize DeCoVec computer
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            demonstration_sampler: ICL example selector
            device: Device
            dataset: Dataset type
        """
        self.model = model
        self.tokenizer = tokenizer
        self.demonstration_sampler = demonstration_sampler
        self.device = device
        self.dataset = dataset
        
        # Create steering computer
        self.steering_computer = TaskVectorBuilder()
    
    def compute_delta_z_with_centering(
        self,
        calibration_data: List[Dict],
        knn_index: Any
    ) -> Dict:
        """
        Compute δz (based on centered logits)
        
        Core steps:
        1. For each calibration sample, compute zero-shot and ICL logits
        2. Center the logits from both models
        3. Compute δz = logits_icl_centered - logits_zero_shot_centered
        
        Args:
            calibration_data: Calibration set data
            knn_index: kNN index
        
        Returns:
            delta_z_cache: Contains δz and related information
        """
        print("\n" + "=" * 80)
        print("Computing δz (on calibration set)")
        print("=" * 80)
        
        delta_z_list = []  # Store δz for all samples
        sample_info = []   # Store sample info (for subsequent evaluation)
        
        print(f"Processing {len(calibration_data)} calibration samples...")
        
        for idx, item in tqdm(enumerate(calibration_data), total=len(calibration_data), desc="Computing δz"):
            
            # Get answer (based on dataset type)
            if self.dataset in ["aqua_rat","aqua_rat_pool_ablation"]:
                best_answer = item["answer"]
            elif self.dataset in ["math500", "math500_pool_ablation"]:
                best_answer = item["final_answer"]
            else:
                best_answer = item["best_answer"]
            
            # 1. Get ICL examples (calibration set uses real-time computation, as neighbors include self)
            examples = self.demonstration_sampler.get_icl_examples(
                idx, 
                calibration_data, 
                use_precomputed=False  # Calibration samples don't use precomputed neighbors
            )
            
            # 2. Construct two types of prompts
            # Zero-shot prompt (original)
            zero_shot_prompt = DemonstrationSampler.construct_zero_shot_prompt(
                item,
                include_answer=True,
                answer=best_answer,
                dataset_type=self.dataset
            )
            
            # ICL prompt (enhanced)
            icl_prompt = self.demonstration_sampler.construct_icl_prompt(
                examples,
                item,
                include_query_answer=True,
                query_answer=best_answer
            )
            
            # 3. Use standard tokenize_prompt_and_continuation to get answer tokens
            # This correctly handles spaces, ensuring answer token is ' true' not 'true'
            # Zero-shot
            zero_shot_prompt_only = DemonstrationSampler.construct_zero_shot_prompt(
                item,
                dataset_type=self.dataset
            )
            _, zero_shot_answer_ids = tokenize_prompt_and_continuation(
                self.tokenizer, zero_shot_prompt_only, best_answer
            )
            
            # ICL
            icl_prompt_only = self.demonstration_sampler.construct_icl_prompt(examples, item)
            _, icl_answer_ids = tokenize_prompt_and_continuation(
                self.tokenizer, icl_prompt_only, best_answer
            )
            
            # 4. Check if answer tokens are valid
            if len(zero_shot_answer_ids) == 0 or len(icl_answer_ids) == 0:
                continue
            
            # Use the first answer token as calibration target
            correct_token = zero_shot_answer_ids[0]
            
            # Construct complete input (for getting logits)
            zero_shot_full_ids = self.tokenizer.encode(zero_shot_prompt, add_special_tokens=True)
            icl_full_ids = self.tokenizer.encode(icl_prompt, add_special_tokens=True)
            
            # 5. Compute logits
            with torch.no_grad():
                # Zero-shot logits
                zero_shot_inputs = torch.tensor([zero_shot_full_ids]).to(self.device)
                zero_shot_outputs = self.model(zero_shot_inputs)
                zero_shot_logits = zero_shot_outputs.logits[0]
                
                # ICL logits  
                icl_inputs = torch.tensor([icl_full_ids]).to(self.device)
                icl_outputs = self.model(icl_inputs)
                icl_logits = icl_outputs.logits[0]
                
                # 6. Compute answer start position (position of last prompt token)
                # The model's output at this position predicts the first answer token
                zero_shot_prompt_len = len(zero_shot_full_ids) - len(zero_shot_answer_ids)
                icl_prompt_len = len(icl_full_ids) - len(icl_answer_ids)
                
                # Answer prediction position (last token of prompt)
                zero_shot_pos = zero_shot_prompt_len - 1
                icl_pos = icl_prompt_len - 1
                
                if zero_shot_pos < 0 or icl_pos < 0:
                    continue
                
                zero_shot_logits_at_pos = zero_shot_logits[zero_shot_pos]
                icl_logits_at_pos = icl_logits[icl_pos]
                
                # 7. Use unified compute_delta_z interface to compute δz
                delta_z = self.steering_computer.compute_delta_z(
                    zero_shot_logits_at_pos,
                    icl_logits_at_pos
                )
                
                delta_z_list.append(delta_z.cpu().numpy())
                
                # Save sample info
                sample_info.append({
                    "idx": idx,
                    "question": item["question"],
                    "best_answer": best_answer,
                    "correct_token": correct_token,
                    "zero_shot_prompt": zero_shot_prompt,
                    "icl_prompt": icl_prompt,
                    "zero_shot_pos": zero_shot_pos,
                    "icl_pos": icl_pos,
                    "icl_logits": icl_logits_at_pos.cpu(),
                    "zero_shot_logits": zero_shot_logits_at_pos.cpu()
                })
        
        print(f"✓ Successfully processed {len(delta_z_list)} samples")
        
        # 8. Aggregate δz
        print("\nAggregating δz statistics...")
        delta_z_array = np.array(delta_z_list)  # [N, vocab_size]
        
        # Compute mean of δz (applied during evaluation)
        delta_z_mean = np.mean(delta_z_array, axis=0)  # [vocab_size]
        
        print(f"  δz shape: {delta_z_array.shape}")
        print(f"  δz mean range: [{delta_z_mean.min():.4f}, {delta_z_mean.max():.4f}]")
        print(f"  δz mean norm: {np.linalg.norm(delta_z_mean):.4f}")
        print(f"  δz std: {np.std(delta_z_array):.4f}")
        
        # Construct cache
        cache_data = {
            "delta_z_array": delta_z_array,
            "delta_z_mean": delta_z_mean,
            "sample_info": sample_info,
            "n_shot": self.demonstration_sampler.n_shot
        }
        
        print("=" * 80)
        
        return cache_data
    
    # calibrate_lambda method removed (lambda is a manual hyperparameter in the paper, not auto-calibrated)
