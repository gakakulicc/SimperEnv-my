from typing import Optional, Sequence
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import cv2 as cv
from .pi0_model import PI0Model
from .pi0_config import PI0Config

class PI0Inference:
    def __init__(self, config: PI0Config):
        self.config = config
        self.model = PI0Model(config)
        self.task_description = None
        self.num_image_history = 0

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        raw_actions = self.model.predict_action(image, self.task_description)
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.config.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.config.action_scale
        action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        action["terminate_episode"] = np.array([0.0])

        return raw_action, action