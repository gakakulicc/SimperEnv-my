from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class PI0Config:
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Normalization
    normalization_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )

    # Image preprocessing
    resize_imgs_with_padding: List[int] = field(default_factory=lambda: [224, 224])

    # Policy setup
    policy_setup: str = "widowx_bridge"
    horizon: int = 1
    pred_action_horizon: int = 1
    exec_horizon: int = 1
    action_scale: float = 1.0

    # Model paths
    saved_model_path: str = "lerobot/pi0"
    unnorm_key: Optional[str] = None