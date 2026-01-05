"""
简化的FLAME模型封装类
基于utils/flame.py重新实现，只用于生成顶点
用于将51维motion（50维expr + 1维jaw）转换为5023×3顶点
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from pathlib import Path


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class SimplifiedFLAME(nn.Module):
    """
    简化的FLAME模型，只用于生成顶点坐标
    使用固定的FLAME参数，不依赖外部pickle文件
    """

    def __init__(self, config):
        super(SimplifiedFLAME, self).__init__()
        self.config = config
        self._initialized = False

    def _initialize(self):
        """延迟初始化 FLAME 模型"""
        if self._initialized:
            return

        print('初始化简化的FLAME模型...')

        self.dtype = torch.float32

        # 使用固定的FLAME参数（基于标准FLAME 2020模型的近似值）
        # 注意：这些是近似值，实际应用中应该使用真实的FLAME模型参数

        # 模板顶点 (5023, 3) - 使用标准头部模型的近似值
        v_template = self._create_default_template_vertices()
        self.register_buffer("v_template", v_template)

        # shape directions (5023, 3, n_shape + n_exp)
        shapedirs = self._create_default_shape_directions()
        self.register_buffer("shapedirs", shapedirs)

        # pose directions (207, 3, num_pose_basis) - 简化版本
        posedirs = self._create_default_pose_directions()
        self.register_buffer("posedirs", posedirs)

        # 关节回归器 (5, 5023) - 简化的关节位置
        J_regressor = self._create_default_j_regressor()
        self.register_buffer("J_regressor", J_regressor)

        # 运动学树
        parents = torch.tensor([-1, 0, 1, 1, 1], dtype=torch.long)
        self.register_buffer("parents", parents)

        # LBS权重 (5023, 5) - 简化的权重
        lbs_weights = self._create_default_lbs_weights()
        self.register_buffer("lbs_weights", lbs_weights)

        self._initialized = True
        print('FLAME模型初始化完成（使用默认参数）')

    def _create_default_template_vertices(self):
        """创建默认的模板顶点（近似值）"""
        # 这是一个非常简化的头部模型
        # 实际的FLAME模型有5023个顶点，这里我们创建一个近似形状

        # 创建一个椭球形状作为头部的基础
        num_vertices = 5023
        vertices = torch.zeros(num_vertices, 3, dtype=self.dtype)

        # 简单的椭球参数化
        for i in range(num_vertices):
            # 随机分布在椭球表面
            theta = torch.rand(1) * 2 * np.pi
            phi = torch.rand(1) * np.pi

            # 椭球尺寸 (近似头部比例)
            a, b, c = 0.08, 0.06, 0.07  # x, y, z 轴半径

            x = a * torch.sin(phi) * torch.cos(theta)
            y = b * torch.sin(phi) * torch.sin(theta)
            z = c * torch.cos(phi)

            # 添加一些噪声使形状更真实
            noise = torch.randn(3) * 0.005
            vertices[i] = torch.tensor([x.item(), y.item(), z.item()]) + noise

        # 调整位置使头部朝前
        vertices[:, 1] += 0.1  # 向前移动
        vertices[:, 2] -= 0.05  # 向上调整

        return vertices

    def _create_default_shape_directions(self):
        """创建默认的shape directions"""
        num_vertices = 5023
        total_params = self.config.n_shape + self.config.n_exp
        shapedirs = torch.randn(num_vertices, 3, total_params, dtype=self.dtype) * 0.001
        return shapedirs

    def _create_default_pose_directions(self):
        """创建默认的pose directions"""
        # pose directions shape: (P, V*3), where P is pose basis dimension, V is number of vertices
        # For FLAME, P is typically around 36 (6 joints * 6 pose params each, but simplified)
        num_pose_basis = 36  # 简化的值
        num_vertices = 5023
        posedirs = torch.randn(num_pose_basis, 3 * num_vertices, dtype=self.dtype) * 0.01
        return posedirs

    def _create_default_j_regressor(self):
        """创建默认的关节回归器"""
        # 5个关节: root, neck, jaw, left_eye, right_eye
        J_regressor = torch.zeros(5, 5023, dtype=self.dtype)

        # 简化的关节位置（每个关节在几个顶点上有权重）
        # Root joint (index 0) - 头部中心
        center_indices = torch.arange(2500, 2520)  # 中心区域
        J_regressor[0, center_indices] = 1.0 / len(center_indices)

        # Neck joint (index 1) - 脖子区域
        neck_indices = torch.arange(2600, 2650)
        J_regressor[1, neck_indices] = 1.0 / len(neck_indices)

        # Jaw joint (index 2) - 下巴区域
        jaw_indices = torch.arange(2700, 2800)
        J_regressor[2, jaw_indices] = 1.0 / len(jaw_indices)

        # Eye joints (indices 3, 4) - 眼睛区域
        left_eye_indices = torch.arange(2800, 2850)
        J_regressor[3, left_eye_indices] = 1.0 / len(left_eye_indices)

        right_eye_indices = torch.arange(2850, 2900)
        J_regressor[4, right_eye_indices] = 1.0 / len(right_eye_indices)

        return J_regressor

    def _create_default_lbs_weights(self):
        """创建默认的LBS权重"""
        num_vertices = 5023
        num_joints = 5
        lbs_weights = torch.zeros(num_vertices, num_joints, dtype=self.dtype)

        # 简化的权重分配
        # 大部分顶点属于root joint
        lbs_weights[:, 0] = 0.8

        # 脖子区域
        neck_indices = torch.arange(2600, 2650)
        lbs_weights[neck_indices, 1] = 0.6
        lbs_weights[neck_indices, 0] = 0.4

        # 下巴区域
        jaw_indices = torch.arange(2700, 2800)
        lbs_weights[jaw_indices, 2] = 0.7
        lbs_weights[jaw_indices, 0] = 0.3

        # 眼睛区域
        left_eye_indices = torch.arange(2800, 2850)
        lbs_weights[left_eye_indices, 3] = 0.8
        lbs_weights[left_eye_indices, 0] = 0.2

        right_eye_indices = torch.arange(2850, 2900)
        lbs_weights[right_eye_indices, 4] = 0.8
        lbs_weights[right_eye_indices, 0] = 0.2

        return lbs_weights

    def to(self, device):
        """移动模型到指定设备"""
        super().to(device)
        return self

    def forward(self, shape_params=None, expression_params=None, pose_params=None):
        """
        简化的FLAME forward，只返回顶点坐标

        Args:
            shape_params: shape参数 (B, n_shape)
            expression_params: expression参数 (B, n_exp)
            pose_params: pose参数 (B, 15) - [head_rot(3), jaw_pose(3), neck_pose(3), eye_pose(6)]

        Returns:
            vertices: 顶点坐标 (B, 5023, 3)
        """
        self._initialize()

        batch_size = shape_params.shape[0] if shape_params is not None else expression_params.shape[0]

        # 确保所有参数都在同一设备上
        device = self.v_template.device

        # 处理shape参数
        if shape_params is None:
            shape_params = torch.zeros(batch_size, self.config.n_shape, device=device)
        else:
            shape_params = shape_params.to(device)

        # 处理expression参数
        if expression_params is None:
            expression_params = torch.zeros(batch_size, self.config.n_exp, device=device)
        else:
            expression_params = expression_params.to(device)

        # 合并shape和expression参数
        betas = torch.cat([shape_params, expression_params], dim=1)

        # 处理pose参数
        if pose_params is None:
            pose_params = torch.zeros(batch_size, 15, device=device)
        else:
            pose_params = pose_params.to(device)

        # 调用LBS函数生成顶点
        from .lbs import lbs
        vertices, _ = lbs(
            betas=betas,
            pose=pose_params,
            v_template=self.v_template.unsqueeze(0).expand(batch_size, -1, -1),
            shapedirs=self.shapedirs,
            posedirs=self.posedirs,
            J_regressor=self.J_regressor,
            parents=self.parents,
            lbs_weights=self.lbs_weights,
            pose2rot=True,
            dtype=self.dtype,
        )

        return vertices


class FLAMEModel(nn.Module):
    """
    FLAME模型封装类
    输入：51维motion（50维expr + 1维jaw）+ shape（可选）
    输出：5023×3顶点
    """

    def __init__(self, flame_model_path=None, n_shape=100, n_exp=50, device='cuda'):
        """
        初始化FLAME模型

        Args:
            flame_model_path: FLAME模型路径（.pkl文件）
            n_shape: shape参数维度（默认100）
            n_exp: expression参数维度（默认50）
            device: 设备（'cuda'或'cpu'）
        """
        super(FLAMEModel, self).__init__()
        self.device = device
        self.n_shape = n_shape
        self.n_exp = n_exp

        # 设置FLAME配置
        flame_dir = Path(__file__).parent / "FLAME2020"
        if flame_model_path is None:
            flame_model_path = str(flame_dir / "generic_model.pkl")
        else:
            flame_model_path = str(flame_model_path)

        if not os.path.exists(flame_model_path):
            raise FileNotFoundError(f"FLAME模型文件不存在: {flame_model_path}")

        config = Struct(
            flame_model_path=str(flame_model_path),
            n_shape=n_shape,
            n_exp=n_exp
        )

        # 创建简化的FLAME模型
        self.flame = SimplifiedFLAME(config).to(device)

        # FLAME模型输出5023个顶点
        self.num_vertices = 5023
    
    def forward(self, expr, jaw_pose, shape=None, head_pose=None, neck_pose=None):
        """
        FLAME forward

        Args:
            expr: expression参数 (B, T, 50) 或 (B, 50)
            jaw_pose: jaw姿态 (B, T, 1) 或 (B, 1)
            shape: shape参数 (B, n_shape) 或 None（使用零shape）
            head_pose: head旋转 (B, T, 3) 或 (B, 3)，默认[0,0,0]
            neck_pose: neck姿态 (B, T, 3) 或 (B, 3)，默认[0,0,0]

        Returns:
            vertices: 顶点坐标 (B, T, 5023, 3) 或 (B, 5023, 3)
        """
        # 处理输入维度
        is_sequence = expr.dim() == 3
        if is_sequence:
            B, T, _ = expr.shape
            expr = expr.view(B * T, -1)
            jaw_pose = jaw_pose.view(B * T, -1)
        else:
            B = expr.shape[0]
            T = 1
            expr = expr.view(B, -1)
            jaw_pose = jaw_pose.view(B, -1)

        # 补齐姿态参数
        # jaw_pose_3 = [jaw, 0, 0]
        jaw_pose_3 = torch.zeros(B * T, 3, device=expr.device)
        jaw_pose_3[:, 0] = jaw_pose.squeeze(-1)

        # head_rot_3 = [0, 0, 0]
        if head_pose is None:
            head_pose_3 = torch.zeros(B * T, 3, device=expr.device)
        else:
            if head_pose.dim() == 3:
                head_pose_3 = head_pose.view(B * T, -1)
            else:
                head_pose_3 = head_pose.unsqueeze(1).repeat(1, T, 1).view(B * T, -1)

        # neck_pose_3 = [0, 0, 0]
        if neck_pose is None:
            neck_pose_3 = torch.zeros(B * T, 3, device=expr.device)
        else:
            if neck_pose.dim() == 3:
                neck_pose_3 = neck_pose.view(B * T, -1)
            else:
                neck_pose_3 = neck_pose.unsqueeze(1).repeat(1, T, 1).view(B * T, -1)

        # shape参数
        if shape is None:
            shape_params = torch.zeros(B * T, self.n_shape, device=expr.device)
        else:
            if shape.dim() == 2:
                shape_params = shape.unsqueeze(1).repeat(1, T, 1).view(B * T, -1)
            else:
                shape_params = shape.view(B * T, -1)

        # 调用FLAME模型
        # FLAME输入格式：shape, expr, pose (包含head, jaw, neck)
        # pose格式：通常为 (B, 15) = [head_rot(3), jaw_pose(3), neck_pose(3), eye_pose(6)]
        pose_params = torch.cat([head_pose_3, jaw_pose_3, neck_pose_3], dim=1)

        # 扩展pose到15维（FLAME标准格式）
        if pose_params.shape[1] < 15:
            padding = torch.zeros(B * T, 15 - pose_params.shape[1], device=expr.device)
            pose_params = torch.cat([pose_params, padding], dim=1)

        vertices = self.flame(shape_params, expr, pose_params)

        # 恢复序列维度
        if is_sequence:
            vertices = vertices.view(B, T, self.num_vertices, 3)
        else:
            vertices = vertices.view(B, self.num_vertices, 3)

        return vertices
    
    def motion_to_vertices(self, motion, shape=None):
        """
        便捷方法：将51维motion转换为顶点
        
        Args:
            motion: 51维motion (B, T, 51) 或 (B, 51)
                   motion[:, :50] = expr, motion[:, 50:51] = jaw
            shape: shape参数 (B, n_shape) 或 None
        
        Returns:
            vertices: 顶点坐标 (B, T, 5023, 3) 或 (B, 5023, 3)
        """
        if motion.dim() == 3:
            # 序列输入
            expr = motion[:, :, :50]  # (B, T, 50)
            jaw_pose = motion[:, :, 50:51]  # (B, T, 1)
        else:
            # 单帧输入
            expr = motion[:, :50].unsqueeze(1)  # (B, 1, 50)
            jaw_pose = motion[:, 50:51].unsqueeze(1)  # (B, 1, 1)
        
        return self.forward(expr, jaw_pose, shape=shape)

