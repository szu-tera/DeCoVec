"""
Evaluator Factory Module
Unified creation of different types of evaluators
"""
from evaluate.baseline_evaluator import BaselineEvaluator
from decovec.demonstration_sampler import DemonstrationSampler


class EvaluatorFactory:
    """Evaluator Factory"""
    
    @staticmethod
    def create_baseline_evaluator(
        model,
        tokenizer,
        device: str,
        demonstration_sampler: DemonstrationSampler,
        dataset: str = "truthfulqa",
        batch_size: int = 8,
        temperature: float = 0.7,
        save_outputs: bool = False,
        output_dir: str = "results/case",
        model_name: str = None,
        self_consistency: bool = False,
        run_id: int = None
    ) -> BaselineEvaluator:
        """
        Create baseline evaluator
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            device: Device
            demonstration_sampler: ICL example selector
            dataset: Dataset type
            batch_size: Batch size
            temperature: Generation temperature (for generative tasks)
            save_outputs: Whether to save model outputs for generative datasets
            output_dir: Output file save directory
            model_name: Model name (for filename generation)
            self_consistency: Whether to enable Self-Consistency mode (uses temperature=0.7 sampling when enabled)
            run_id: Run ID (for filename in Self-Consistency mode)
        
        Returns:
            Baseline evaluator instance
        """
        return BaselineEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            demonstration_sampler=demonstration_sampler,
            dataset=dataset,
            batch_size=batch_size,
            temperature=temperature,
            save_outputs=save_outputs,
            output_dir=output_dir,
            model_name=model_name,
            self_consistency=self_consistency,
            run_id=run_id
        )

