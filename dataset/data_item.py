"""
数据集单个样本处理类
处理FLAME系数格式差异、帧率统一等
"""
import os
import numpy as np
import torch
from scipy.interpolate import interp1d
import json


class DataItem:
    """
    处理单个数据样本
    支持digital_human、MEAD_VHAP、MultiModal200三种数据集格式
    """
    
    def __init__(self, sample_id, sample_info, data_root, flame_model=None):
        """
        初始化数据样本
        
        Args:
            sample_id: 样本ID
            sample_info: JSON中的样本信息字典
            data_root: 数据根目录（用于拼接相对路径）
            flame_model: FLAME模型实例（用于生成GT顶点）
        """
        self.sample_id = sample_id
        self.sample_info = sample_info
        self.data_root = data_root
        self.flame_model = flame_model
        
        # 从sample_info中提取信息
        self.audio_path = sample_info.get('audio_path', '')
        self.flame_path = sample_info.get('flame_coeff_save_path') or sample_info.get('flame_path', '')
        
        # speaker_id可能是字符串或列表
        speaker_id_raw = sample_info.get('speaker_id', '')
        if isinstance(speaker_id_raw, list):
            self.speaker_id = speaker_id_raw[0] if len(speaker_id_raw) > 0 else ''
        else:
            self.speaker_id = speaker_id_raw
        
        # emotion可能是字符串或从speaker_id列表中提取
        emotion_raw = sample_info.get('emotion', '')
        if isinstance(emotion_raw, list):
            self.emotion = emotion_raw[0] if len(emotion_raw) > 0 else ''
        elif isinstance(speaker_id_raw, list) and len(speaker_id_raw) > 1:
            self.emotion = speaker_id_raw[1]  # 从speaker_id列表中提取emotion
        else:
            self.emotion = emotion_raw
        
        # 检测数据集类型
        self.dataset_name = self._detect_dataset_name()
        
        # 检测帧率
        self.fps = self._detect_fps()
    
    def _detect_dataset_name(self):
        """检测数据集名称"""
        flame_path_lower = self.flame_path.lower()
        if 'multimodal' in flame_path_lower or 'multimodal200' in flame_path_lower:
            return 'multimodal200'
        elif 'mead' in flame_path_lower or 'vhap' in flame_path_lower:
            return 'mead_vhap'
        elif 'digital_human' in flame_path_lower or 'digitalhuman' in flame_path_lower:
            return 'digital_human'
        else:
            # 默认根据路径判断
            if 'multimodal' in self.flame_path.lower():
                return 'multimodal200'
            elif 'mead' in self.flame_path.lower():
                return 'mead_vhap'
            else:
                return 'digital_human'
    
    def _detect_fps(self):
        """检测帧率"""
        # 根据baseline_prompt.md的逻辑
        if 'multimodal' in self.dataset_name.lower():
            return 20
        elif 'mead' in self.dataset_name.lower():
            return 25
        else:
            # digital_human默认为25fps
            return 25
    
    def load_flame_coeffs(self):
        """
        加载FLAME系数，并转换为统一格式
        
        Returns:
            expr: expression参数 (T, 50)
            jaw_pose: jaw姿态 (T, 1)
            shape: shape参数 (T, n_shape) 或 None
        """
        # 根据baseline_prompt.md，data_root统一为/home/caizhuoqiang/Data
        # JSON中的flame_path是相对于data_root的相对路径
        # 例如：flame_path可能是 "MEAD_VHAP/xxx/tracked_flame_params_20.npz" 或 "MultiModal200/xxx/tracked_flame_params_20.npz"
        if os.path.isabs(self.flame_path):
            flame_file = self.flame_path
        else:
            # 直接拼接：data_root + flame_path
            # data_root = /home/caizhuoqiang/Data
            # flame_path = MEAD_VHAP/xxx/tracked_flame_params_20.npz
            # 结果 = /home/caizhuoqiang/Data/MEAD_VHAP/xxx/tracked_flame_params_20.npz
            flame_file = os.path.join(self.data_root, self.flame_path)
        
        if not os.path.exists(flame_file):
            # 如果文件不存在，尝试其他可能的路径格式（向后兼容）
            alt_flame_file = os.path.join('/home/caizhuoqiang/Data', self.flame_path)
            if os.path.exists(alt_flame_file):
                flame_file = alt_flame_file
            else:
                raise FileNotFoundError(f"FLAME file not found: {flame_file} (also tried: {alt_flame_file})")
        
        # 加载npz文件
        data = np.load(flame_file)
        
        if self.dataset_name == 'digital_human':
            # digital_human格式：expcode, posecode (包含head+jaw), shapecode
            expcode = data['expcode']  # (T, 50)
            posecode = data['posecode']  # (T, 6) = head(3) + jaw(3)
            shapecode = data.get('shapecode', None)  # (T, n_shape) 或 (n_shape,)
            
            # 提取jaw_pose（只取第一维）
            jaw_pose = posecode[:, 3:4]  # (T, 1)
            expr = expcode  # (T, 50)
            
            # 处理shapecode
            if shapecode is not None:
                if shapecode.ndim == 1:
                    # 如果是单帧shape，扩展到序列长度
                    shape = np.tile(shapecode[None, :], (expr.shape[0], 1))
                else:
                    shape = shapecode
            else:
                shape = None
        
        elif self.dataset_name in ['mead_vhap', 'multimodal200']:
            # MEAD_VHAP/MultiModal200格式：expr, jaw_pose, shape
            expr = data['expr']  # (T, 50)
            jaw_pose = data['jaw_pose']  # (T, 3) 或 (T, 1)
            shape = data.get('shape', None)  # (T, n_shape) 或 (n_shape,)
            
            # 如果jaw_pose是3维，只取第一维
            if jaw_pose.ndim == 2 and jaw_pose.shape[1] == 3:
                jaw_pose = jaw_pose[:, 0:1]  # (T, 1)
            elif jaw_pose.ndim == 1:
                jaw_pose = jaw_pose[:, None]  # (T, 1)
            
            # 处理shape
            if shape is not None:
                if shape.ndim == 1:
                    shape = np.tile(shape[None, :], (expr.shape[0], 1))
            else:
                shape = None
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return expr, jaw_pose, shape
    
    def resample_to_25fps(self, data, original_fps):
        """
        将数据重采样到25fps
        
        Args:
            data: 数据数组 (T, D)
            original_fps: 原始帧率
        
        Returns:
            resampled_data: 重采样后的数据 (T_new, D)
        """
        if original_fps == 25:
            return data
        
        T, D = data.shape
        original_time = np.linspace(0, T / original_fps, T)
        target_fps = 25
        target_time = np.linspace(0, T / original_fps, int(T * target_fps / original_fps))
        
        resampled_data = np.zeros((len(target_time), D))
        for d in range(D):
            interp_func = interp1d(original_time, data[:, d], kind='linear', fill_value='extrapolate')
            resampled_data[:, d] = interp_func(target_time)
        
        return resampled_data
    
    def get_motion(self, max_length=None):
        """
        获取51维motion（50维expr + 1维jaw）
        
        Args:
            max_length: 最大序列长度（25fps帧数），如果为None则不裁剪
        
        Returns:
            motion: (T, 51) numpy array
        """
        expr, jaw_pose, _ = self.load_flame_coeffs()
        
        # 重采样到25fps
        if self.fps != 25:
            expr = self.resample_to_25fps(expr, self.fps)
            jaw_pose = self.resample_to_25fps(jaw_pose, self.fps)
        
        # 拼接为51维motion
        motion = np.concatenate([expr, jaw_pose], axis=1)  # (T, 51)
        
        # 裁剪长度（如果指定了max_length）
        if max_length is not None and motion.shape[0] > max_length:
            # 从中间裁剪或从开头裁剪
            motion = motion[:max_length]
        
        return motion
    
    def get_vertices(self, shape=None):
        """
        使用FLAME模型生成顶点
        
        Args:
            shape: shape参数 (n_shape,) 或 None
        
        Returns:
            vertices: (T, 5023, 3) numpy array
        """
        if self.flame_model is None:
            raise ValueError("FLAME model is required to generate vertices")
        
        # 获取motion
        motion = self.get_motion()  # (T, 51)
        
        # 转换为tensor
        motion_tensor = torch.FloatTensor(motion).unsqueeze(0)  # (1, T, 51)
        
        # 处理shape
        if shape is not None:
            if shape.ndim == 1:
                shape_tensor = torch.FloatTensor(shape).unsqueeze(0)  # (1, n_shape)
            else:
                shape_tensor = torch.FloatTensor(shape[0]).unsqueeze(0)  # 使用第一帧的shape
        else:
            shape_tensor = None
        
        # FLAME forward
        with torch.no_grad():
            vertices = self.flame_model.motion_to_vertices(motion_tensor, shape=shape_tensor)
            vertices = vertices.squeeze(0).cpu().numpy()  # (T, 5023, 3)
        
        return vertices
    
    def get_audio_path(self):
        """获取音频文件路径（绝对路径）"""
        return os.path.join(self.data_root, self.audio_path)

