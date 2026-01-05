"""
FLAME模型封装类
用于将51维motion（50维expr + 1维jaw）转换为5023×3顶点
"""
import torch
import torch.nn as nn
import numpy as np
import os


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
        
        # 尝试导入FLAME模型
        try:
            from FLAME import FLAME
            self.flame = FLAME(n_shape=n_shape, n_exp=n_exp).to(device)
            if flame_model_path and os.path.exists(flame_model_path):
                # 加载预训练模型权重（如果需要）
                pass
        except ImportError:
            # 如果没有FLAME库，创建一个占位符
            # 实际使用时需要安装flame_pytorch或类似库
            print("Warning: FLAME library not found. Please install flame_pytorch.")
            print("For now, using a placeholder that returns zeros.")
            self.flame = None
        
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
        if self.flame is None:
            # 占位符：返回零顶点
            if expr.dim() == 3:
                B, T, _ = expr.shape
                return torch.zeros(B, T, self.num_vertices, 3, device=expr.device)
            else:
                B, _ = expr.shape
                return torch.zeros(B, self.num_vertices, 3, device=expr.device)
        
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
        # pose格式：通常为 (B, 15) = [head_rot(3), jaw_pose(3), neck_pose(3), ...]
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

