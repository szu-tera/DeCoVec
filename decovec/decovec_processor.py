"""
DeCoVec Logits Processor for Generation Tasks

Applies DeCoVec adjustment to logits during generation for generative tasks.
"""
import torch
from transformers import LogitsProcessor
from typing import Dict, List, Optional

from simple_config import get_config
from decovec.decovec_core import TaskVectorBuilder

class DeCoVecLogitsProcessor(LogitsProcessor):
    """
    Dynamic DeCoVec LogitsProcessor - Real-time delta_z computation and direct logits adjustment with λ·δz
    
    At each token generation position:
    1. Extract already generated tokens
    2. Construct zero-shot and ICL complete prompts
    3. Forward pass on both prompts to get logits
    4. Compute real-time delta_z = compute_delta_z(zero_shot_logits, icl_logits)
    5. Apply DeCoVec adjustment: logits' = logits + λ * delta_z
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        zero_shot_prompt: str,
        icl_prompt: str,
        lambda_scale: float,
        steering_computer: TaskVectorBuilder,
        device: str = "cuda",
        vector_icl_prompt: Optional[str] = None
    ):
        """
        Args:
            model: Language model
            tokenizer: tokenizer
            zero_shot_prompt: Zero-shot prompt (without answer)
            icl_prompt: ICL prompt (for inference, without answer)
            lambda_scale: Global calibration strength λ
            steering_computer: TaskVectorBuilder instance
            device: Compute device
            vector_icl_prompt: ICL prompt for constructing task vector (if None, uses icl_prompt)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.zero_shot_prompt = zero_shot_prompt
        self.icl_prompt = icl_prompt
        self.vector_icl_prompt = vector_icl_prompt if vector_icl_prompt is not None else icl_prompt
        self.lambda_scale = lambda_scale
        self.steering_computer = steering_computer
        self.device = device
        
        # Pre-tokenize original prompts to track length
        self.zero_shot_prompt_ids = tokenizer.encode(zero_shot_prompt, add_special_tokens=True)
        self.icl_prompt_ids = tokenizer.encode(icl_prompt, add_special_tokens=True)
        self.vector_icl_prompt_ids = tokenizer.encode(self.vector_icl_prompt, add_special_tokens=True)
        self.initial_length = len(self.icl_prompt_ids)
        self.zero_shot_prompt_len = len(self.zero_shot_prompt_ids)
        
        # Maintain zero-shot inference cache state to avoid full sequence forward pass at each token
        self.zero_shot_generated_tokens = 0
        self.zero_shot_attention_mask = torch.ones(
            (1, self.zero_shot_prompt_len),
            dtype=torch.long,
            device=self.device
        )
        self.zero_shot_past_key_values = None
        self.last_zero_shot_logits = None
        self._init_zero_shot_cache()
        
        # Maintain vector_icl inference cache state (if different from icl_prompt)
        if self.vector_icl_prompt != self.icl_prompt:
            self._init_vector_icl_cache()
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Apply dynamic DeCoVec adjustment to logits
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Current position logits [batch_size, vocab_size]
        
        Returns:
            adjusted_scores: Adjusted logits [batch_size, vocab_size]
        """
        adjusted_scores = scores.clone()
        
        # When λ≈0, return original logits directly to avoid delta_z computation (may contain -inf, causing 0 * (-inf) = nan)
        # Note: Even without errors, nan will cause log_softmax to produce abnormal distributions, affecting generation quality
        if abs(self.lambda_scale) < 1e-10:
            return adjusted_scores
        
        for i in range(scores.shape[0]):
            # 1. Extract generated tokens (beyond original prompt)
            # Optimization: Avoid CPU-GPU communication from .tolist(), operate directly on GPU
            current_seq_len = input_ids[i].shape[0]
            
            # If still within original prompt, don't apply steering (theoretically shouldn't happen)
            if current_seq_len <= self.initial_length:
                continue
            
            # Extract generated portion (operate on GPU, avoid CPU-GPU communication)
            generated_ids_tensor = input_ids[i, self.initial_length:]
            generated_len = generated_ids_tensor.shape[0]
            
            if generated_len == 0:
                continue
            
            # Use cached zero-shot state for incremental logits computation
            # Optimization: Pass tensor instead of list to avoid CPU-GPU communication
            zero_shot_logits = self._get_zero_shot_logits_from_tensor(generated_ids_tensor, generated_len)
            icl_logits = scores[i]  # ICL logits for inference
            
            # If using different vector_icl_prompt, need to compute vector_icl_logits
            if self.vector_icl_prompt != self.icl_prompt:
                # Need to compute vector_icl_prompt logits at current generation position
                vector_icl_logits = self._get_vector_icl_logits(generated_ids_tensor, generated_len)
            else:
                vector_icl_logits = icl_logits
            
            # Compute real-time delta_z (using zero-shot and vector_icl logits)
            delta_z = self.steering_computer.compute_delta_z(
                zero_shot_logits,
                vector_icl_logits  # Use vector_icl_method logits
            )
            
            # Directly add λ·δz to current logits (ICL logits for inference)
            # Note: Even if delta_z contains -inf, as long as λ≠0, result is still valid (-inf will be preserved)
            adjusted_scores[i] = icl_logits + self.lambda_scale * delta_z
        
        return adjusted_scores

    def _init_zero_shot_cache(self):
        """Initialize zero-shot prompt cache (past_key_values and logits)"""
        zero_shot_input = torch.tensor([self.zero_shot_prompt_ids]).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                zero_shot_input,
                attention_mask=self.zero_shot_attention_mask,
                use_cache=True,
                return_dict=True
            )
        self.zero_shot_past_key_values = outputs.past_key_values
        self.last_zero_shot_logits = outputs.logits[0, -1, :]
    
    def _get_zero_shot_logits_from_tensor(self, generated_ids_tensor: torch.Tensor, generated_len: int) -> torch.Tensor:
        """
        Incremental zero-shot logits computation: Only one forward pass for newly generated tokens
        Optimized version: Use GPU tensor directly to avoid CPU-GPU communication
        
        Args:
            generated_ids_tensor: Generated token IDs tensor [seq_len], on GPU
            generated_len: Number of generated tokens
        
        Returns:
            zero-shot logits [vocab_size]
        """
        target_generated = generated_len
        if target_generated == self.zero_shot_generated_tokens:
            return self.last_zero_shot_logits
        if target_generated < self.zero_shot_generated_tokens:
            raise ValueError("generated_ids length decreased, cannot sync zero-shot state")
        
        # Optimization: Slice directly on GPU to avoid CPU-GPU communication
        new_tokens_tensor = generated_ids_tensor[self.zero_shot_generated_tokens:].unsqueeze(0)
        new_tokens_count = new_tokens_tensor.shape[1]
        
        new_attention = torch.ones(
            (1, new_tokens_count),
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.cat([self.zero_shot_attention_mask, new_attention], dim=1)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_tokens_tensor,
                attention_mask=attention_mask,
                past_key_values=self.zero_shot_past_key_values,
                use_cache=True,
                return_dict=True
            )
        
        self.zero_shot_attention_mask = attention_mask
        self.zero_shot_past_key_values = outputs.past_key_values
        self.zero_shot_generated_tokens = target_generated
        self.last_zero_shot_logits = outputs.logits[0, -1, :]
        return self.last_zero_shot_logits
    
    def _init_vector_icl_cache(self):
        """Initialize vector_icl prompt cache (past_key_values and logits)"""
        if self.vector_icl_prompt == self.icl_prompt:
            # If same, no need for separate cache
            self.vector_icl_past_key_values = None
            self.vector_icl_attention_mask = None
            self.vector_icl_generated_tokens = 0
            self.last_vector_icl_logits = None
            return
        
        vector_icl_input = torch.tensor([self.vector_icl_prompt_ids]).to(self.device)
        vector_icl_attention_mask = torch.ones(
            (1, len(self.vector_icl_prompt_ids)),
            dtype=torch.long,
            device=self.device
        )
        with torch.no_grad():
            outputs = self.model(
                vector_icl_input,
                attention_mask=vector_icl_attention_mask,
                use_cache=True,
                return_dict=True
            )
        self.vector_icl_past_key_values = outputs.past_key_values
        self.vector_icl_attention_mask = vector_icl_attention_mask
        self.vector_icl_generated_tokens = 0
        self.last_vector_icl_logits = outputs.logits[0, -1, :]
    
    def _get_vector_icl_logits(self, generated_ids_tensor: torch.Tensor, generated_len: int) -> torch.Tensor:
        """
        Incremental vector_icl logits computation: Only one forward pass for newly generated tokens
        
        Args:
            generated_ids_tensor: Generated token IDs tensor [seq_len], on GPU
            generated_len: Number of generated tokens
        
        Returns:
            vector_icl logits [vocab_size]
        """
        if self.vector_icl_prompt == self.icl_prompt:
            # If same, directly return current ICL logits (already obtained in __call__)
            # But here we need to recompute, as we need vector_icl_prompt's logits
            # Actually, if same, this method shouldn't be called
            raise ValueError("vector_icl_prompt is same as icl_prompt, this method should not be called")
        
        if self.vector_icl_past_key_values is None:
            self._init_vector_icl_cache()
        
        target_generated = generated_len
        if target_generated == self.vector_icl_generated_tokens:
            return self.last_vector_icl_logits
        if target_generated < self.vector_icl_generated_tokens:
            raise ValueError("generated_ids length decreased, cannot sync vector_icl state")
        
        # Optimization: Slice directly on GPU to avoid CPU-GPU communication
        new_tokens_tensor = generated_ids_tensor[self.vector_icl_generated_tokens:].unsqueeze(0)
        new_tokens_count = new_tokens_tensor.shape[1]
        
        new_attention = torch.ones(
            (1, new_tokens_count),
            dtype=torch.long,
            device=self.device
        )
        attention_mask = torch.cat([self.vector_icl_attention_mask, new_attention], dim=1)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_tokens_tensor,
                attention_mask=attention_mask,
                past_key_values=self.vector_icl_past_key_values,
                use_cache=True,
                return_dict=True
            )
        
        self.vector_icl_attention_mask = attention_mask
        self.vector_icl_past_key_values = outputs.past_key_values
        self.vector_icl_generated_tokens = target_generated
        self.last_vector_icl_logits = outputs.logits[0, -1, :]
        return self.last_vector_icl_logits

