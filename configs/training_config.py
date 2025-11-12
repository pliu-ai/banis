"""Training configuration classes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training configuration.
    
    Attributes:
        # Experiment settings
        exp_name: Experiment name (auto-generated if empty).
        save_path: Path to save checkpoints and logs.
        seed: Random seed for reproducibility.
        
        # Training settings
        batch_size: Training batch size.
        n_steps: Total number of training steps.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        
        # Learning rate schedule
        use_scheduler: Whether to use learning rate scheduler.
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau').
        
        # Validation
        val_check_interval: Steps between validation checks.
        limit_val_batches: Maximum number of validation batches.
        
        # Logging
        log_every_n_steps: Steps between logging.
        
        # Gradient
        gradient_clip_val: Maximum gradient norm for clipping.
        accumulate_grad_batches: Number of batches for gradient accumulation.
        
        # Precision
        precision: Training precision ('32', '16-mixed', 'bf16-mixed').
        
        # Distributed training
        distributed: Whether to use distributed training.
        devices: Number of GPU devices to use (-1 for all).
        num_nodes: Number of nodes for distributed training.
        
        # Checkpointing
        save_top_k: Number of best checkpoints to save.
        save_last: Whether to save last checkpoint.
        resume_from_checkpoint: Path to checkpoint to resume from.
        
        # Advanced settings
        compile_model: Whether to compile model with torch.compile.
        detect_anomaly: Whether to enable anomaly detection (for debugging).
        profiler: Profiler type ('simple', 'advanced', None).
        fast_dev_run: Number of steps for fast development run (0 to disable).
        
        # External validation (for long training jobs)
        validate_extern: Whether to use external validation process.
        auto_resubmit: Automatically resubmit job on time limit.
    """
    
    # Experiment settings
    exp_name: str = ""
    save_path: Path = Path("./outputs")
    seed: int = 0
    
    # Training settings
    batch_size: int = 8
    n_steps: int = 50_000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    
    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    
    # Validation
    val_check_interval: int = 5000
    limit_val_batches: int = 100
    
    # Logging
    log_every_n_steps: int = 100
    
    # Gradient
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Precision
    precision: str = "16-mixed"
    
    # Distributed training
    distributed: bool = False
    devices: int = -1
    num_nodes: int = 1
    
    # Checkpointing
    save_top_k: int = 3
    save_last: bool = True
    resume_from_checkpoint: Optional[Path] = None
    
    # Advanced settings
    compile_model: bool = True
    detect_anomaly: bool = False
    profiler: Optional[str] = "simple"
    fast_dev_run: int = 0
    
    # External validation
    validate_extern: bool = False
    auto_resubmit: bool = False
    
    def __post_init__(self):
        """Validate and convert paths."""
        if isinstance(self.save_path, str):
            self.save_path = Path(self.save_path)
        if self.resume_from_checkpoint is not None and isinstance(self.resume_from_checkpoint, str):
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)
    
    def validate(self) -> None:
        """Validate training configuration.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if not 0 <= self.weight_decay <= 1:
            raise ValueError(f"weight_decay must be in [0, 1], got {self.weight_decay}")
        
        if self.scheduler_type not in ["cosine", "step", "plateau", "none"]:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")
        
        if self.precision not in ["32", "16-mixed", "bf16-mixed"]:
            raise ValueError(f"Invalid precision: {self.precision}")
        
        if self.profiler not in ["simple", "advanced", None]:
            raise ValueError(f"Invalid profiler: {self.profiler}")

