"""
DeCoVec Core Algorithm Implementation

Implements paper Equation (7): v_T^t = z_icl^t - z_zs^t
"""
import torch


class TaskVectorBuilder:
    """
    Task Vector Calculator
    
    Implements task vector construction from Section 3.2 of the paper.
    Equation (7): v_T^t = z_icl^t - z_zs^t
    """
    
    def __init__(self):
        pass
    
    def compute_delta_z(
        self,
        logits_base: torch.Tensor,
        logits_finetuned: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the task vector (Task Vector)
        
        Implements paper Equation (7): v_T^t = z_icl^t - z_zs^t
        Uses centered logits difference as the steering vector.
        
        Steps:
        1. Center both model logits (subtract their respective means)
        2. Compute the difference of centered logits
        
        Args:
            logits_base: Zero-shot logits z_zs [vocab_size] or [batch, vocab_size]
            logits_finetuned: ICL logits z_icl [vocab_size] or [batch, vocab_size]
        
        Returns:
            delta_z: Task vector v_T [vocab_size] or [batch, vocab_size]
        """
        # Compute mean over vocabulary dimension
        dim = -1
        
        # Centering: subtract respective means
        logits_base_centered = logits_base - logits_base.mean(dim=dim, keepdim=True)
        logits_finetuned_centered = logits_finetuned - logits_finetuned.mean(dim=dim, keepdim=True)
        
        # Compute centered logits difference
        delta_z = logits_finetuned_centered - logits_base_centered
        
        return delta_z

