"""Model configuration classes."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration.
    
    Attributes:
        # Architecture
        architecture: Model architecture name ('mednext').
        model_id: Model size identifier ('S', 'M', 'L').
        kernel_size: Convolutional kernel size.
        num_input_channels: Number of input channels (auto-determined from data).
        num_output_channels: Number of output channels (6 for short+long range affinities).
        
        # SDT (Signed Distance Transform) prediction
        predict_sdt: Whether to predict SDT as additional output.
        sdt_loss_weight: Weight for SDT loss.
        
        # Evaluation
        eval_thresholds: List of thresholds for evaluation.
    """
    
    # Architecture
    architecture: str = "mednext"
    model_id: str = "S"  # S, M, L
    kernel_size: int = 3
    num_input_channels: int = 1  # Auto-determined from data
    num_output_channels: int = 6  # 3 short + 3 long range affinities
    
    # SDT prediction
    predict_sdt: bool = False
    sdt_loss_weight: float = 1.0
    
    # Evaluation
    eval_thresholds: List[float] = field(default_factory=lambda: [
        0.4502, 0.4741, 0.4966, 0.5182, 0.5388,
        0.5584, 0.5769, 0.5945, 0.6111, 0.6267,
        0.6413, 0.6551, 0.6680
    ])
    
    def validate(self) -> None:
        """Validate model configuration.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        if self.architecture not in ["mednext"]:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        
        if self.model_id not in ["S", "M", "L"]:
            raise ValueError(f"Invalid model_id: {self.model_id}, must be one of ['S', 'M', 'L']")
        
        if self.kernel_size not in [3, 5, 7]:
            raise ValueError(f"kernel_size must be 3, 5, or 7, got {self.kernel_size}")
        
        if self.num_input_channels < 1:
            raise ValueError(f"num_input_channels must be positive, got {self.num_input_channels}")
        
        if self.num_output_channels not in [3, 6, 7]:
            raise ValueError(f"num_output_channels must be 3, 6, or 7, got {self.num_output_channels}")
        
        if not all(0 <= t <= 1 for t in self.eval_thresholds):
            raise ValueError("All eval_thresholds must be in [0, 1]")
    
    @property
    def total_output_channels(self) -> int:
        """Get total number of output channels including SDT.
        
        Returns:
            Total number of output channels.
        """
        return self.num_output_channels + (1 if self.predict_sdt else 0)

