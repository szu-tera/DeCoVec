"""
DeCoVec Core Algorithm Implementation

Implements paper Equation (7): v_T^t = z_icl^t - z_zs^t
"""
import torch
import torch.nn.functional as F
from typing import Optional


class TaskVectorBuilder:
    """
    Task Vector Calculator
    
    Implements task vector construction from Section 3.2 of the paper.
    Equation (7): v_T^t = z_icl^t - z_zs^t
    """
    
    def __init__(
        self,
        use_mask: bool = True,
        mask_threshold: float = 0.1,
    ):
        """
        Args:
            use_mask: Whether to mask low-probability tokens in delta_z (default: True)
            mask_threshold: Mask tokens with prob < threshold * max_prob (default: 0.1)
        """
        self.use_mask = use_mask
        self.mask_threshold = mask_threshold
    
    def _compute_delta_z_mask(
        self,
        logits: torch.Tensor,
        mask_threshold: float,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Compute mask for delta_z: tokens with prob < threshold * max_prob are masked (set to 0).
        Uses softmax(logits) to get probabilities.
        
        Args:
            logits: [vocab_size] or [batch, vocab_size]
            mask_threshold: Mask if prob < threshold * max_prob
            device: Device for output mask
        
        Returns:
            mask: 1.0 for keep, 0.0 for mask. Same shape as logits.
        """
        probs = F.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        # keep tokens where prob >= threshold * max_prob
        keep = probs >= (mask_threshold * max_prob)
        return keep.float().to(logits.dtype).to(device or logits.device)
    
    def compute_delta_z(
        self,
        logits_base: torch.Tensor,
        logits_finetuned: torch.Tensor,
        use_mask: Optional[bool] = None,
        mask_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute the task vector (Task Vector)
        
        Implements paper Equation (7): v_T^t = z_icl^t - z_zs^t
        Uses centered logits difference as the steering vector.
        Optionally masks tokens whose probability (after softmax on ICL logits) is
        less than threshold * max_prob; masked positions are set to 0.
        
        Steps:
        1. Center both model logits (subtract their respective means)
        2. Compute the difference of centered logits
        3. Optionally mask: set delta_z[i]=0 where softmax(logits_icl)[i] < threshold * max_prob
        
        Args:
            logits_base: Zero-shot logits z_zs [vocab_size] or [batch, vocab_size]
            logits_finetuned: ICL logits z_icl [vocab_size] or [batch, vocab_size]
            use_mask: Override instance default (None = use self.use_mask)
            mask_threshold: Override instance default (None = use self.mask_threshold)
        
        Returns:
            delta_z: Task vector v_T [vocab_size] or [batch, vocab_size]
        """
        use_mask = use_mask if use_mask is not None else self.use_mask
        mask_threshold = mask_threshold if mask_threshold is not None else self.mask_threshold
        
        # Compute mean over vocabulary dimension
        dim = -1
        
        # Centering: subtract respective means
        logits_base_centered = logits_base - logits_base.mean(dim=dim, keepdim=True)
        logits_finetuned_centered = logits_finetuned - logits_finetuned.mean(dim=dim, keepdim=True)
        
        # Compute centered logits difference
        delta_z = logits_finetuned_centered - logits_base_centered
        
        if use_mask and mask_threshold > 0:
            mask = self._compute_delta_z_mask(logits_finetuned, mask_threshold, device=delta_z.device)
            delta_z = delta_z * mask
        
        return delta_z

