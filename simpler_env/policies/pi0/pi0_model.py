from typing import Optional, Sequence
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForVision2Seq
import torch
from PIL import Image
import cv2

# 嵌入 LeRobotDatasetMetadata 的功能
class LeRobotDatasetMetadata:
    def __init__(self, dummy_dataset: str):
        self.dummy_dataset = dummy_dataset
        self.stats = {}

    def _init_normalization_stats(self):
        """初始化与仿真环境匹配的归一化参数"""
        self.stats = {
            "observation.state": {
                "mean": torch.zeros(32),  # 假设状态维度为32
                "std": torch.ones(32)
            },
            "action": {
                "mean": torch.zeros(7),  # 假设动作维度为7
                "std": torch.ones(7)
            }
        }

# 嵌入 PI0Config 和 PI0Policy 的功能
class PI0Config:
    def __init__(self, model_path: str, policy_setup: str, action_scale: float):
        self.model_path = model_path
        self.policy_setup = policy_setup
        self.action_scale = action_scale

class PI0Policy:
    def __init__(self, model_path: str, device: str):
        self.model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    def select_action(self, batch):
        inputs = self.tokenizer(batch["task"], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs)
        return outputs

class Pi0Inference:
    def __init__(
        self,
        model_path: str = "lerobot/pi0",
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        policy_setup: str = "widowx_bridge",
        device: str = "cuda"
    ) -> None:
        # 初始化 LeRobotDatasetMetadata
        self.dataset_meta = LeRobotDatasetMetadata("dummy_dataset")
        self.dataset_meta._init_normalization_stats()
        
        # 初始化 PI0Config 和 PI0Policy
        self.config = PI0Config(model_path, policy_setup, action_scale)
        self.policy = PI0Policy(model_path, device)
        
        # 图像预处理参数
        self.image_size = image_size
        self.action_scale = action_scale
        self.device = device
        
        # 动作队列管理
        self._action_queue = []
        self.policy_setup = policy_setup

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理流程"""
        # 调整尺寸并转换为Tensor
        image = cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).float() / 255.0  # [0,1]范围
        return image.unsqueeze(0).to(self.device)  # 添加batch维度

    def reset(self, task_description: str) -> None:
        """重置策略状态"""
        self.policy.reset()
        self._action_queue = []
        self.task_description = task_description

    def step(
        self, 
        image: np.ndarray, 
        robot_state: Optional[np.ndarray] = None,
        task_description: Optional[str] = None
    ) -> dict:
        """
        输入：
            image: (H, W, 3) uint8格式的RGB图像
            robot_state: 可选，机器人状态向量
            task_description: 任务描述文本
        输出：
            action_dict: 包含控制指令的字典，格式与仿真环境兼容
        """
        # 1. 图像预处理
        image_tensor = self._preprocess_image(image)
        
        # 2. 构建输入batch
        batch = {
            "observation.images.top": image_tensor,
            "task": [task_description or self.task_description]
        }
        
        if robot_state is not None:
            batch["observation.state"] = torch.from_numpy(robot_state).float().to(self.device).unsqueeze(0)

        # 3. 获取动作（自动处理动作队列）
        if not self._action_queue:
            with torch.no_grad():
                actions = self.policy.select_action(batch)
                self._action_queue = actions.cpu().numpy().squeeze(0)  # (n_action_steps, action_dim)
        
        # 4. 从队列取出当前动作
        raw_action = self._action_queue[0]
        self._action_queue = self._action_queue[1:]

        # 5. 转换为仿真环境需要的格式（示例为WidowX格式）
        action_dict = {
            "world_vector": raw_action[:3] * self.action_scale,     # XYZ位移
            "rot_axangle": raw_action[3:6] * self.action_scale,     # 轴角旋转
            "gripper": [1.0 if raw_action[6] > 0 else -1.0],        # 夹爪开合
            "terminate_episode": [0.0]
        }

        return action_dict

    # def __call__(self, *args, **kwargs):
    #     """保持与OpenVLA相同的调用接口"""
    #     return self.step(*args, **kwargs)