#!/usr/bin/env python3
"""
生成FLAME模板文件
为每个subject创建基准面部网格，作为训练时的参考模板
"""

import os
import pickle
import torch
import numpy as np
from utils.flame_model import FLAMEModel

def create_flame_template(subjects_list, output_path='templates_flame.pkl', device='cuda'):
    """
    为给定的subjects创建FLAME模板

    Args:
        subjects_list: subject ID列表
        output_path: 输出pickle文件路径
        device: 计算设备
    """

    # 初始化FLAME模型
    flame_model = FLAMEModel(device=device)

    # 创建templates字典
    templates = {}

    print(f"为 {len(subjects_list)} 个subjects生成FLAME模板...")

    for subject_id in subjects_list:
        print(f"处理 subject: {subject_id}")

        # 为每个subject生成不同的随机shape参数
        # 使用固定的随机种子确保可重现性
        torch.manual_seed(hash(subject_id) % 2**32)
        np.random.seed(hash(subject_id) % 2**32)

        # 生成shape参数 (100维)
        shape_params = torch.randn(1, 100, device=device) * 0.1  # 小幅度随机shape

        # 生成中性表情的基准网格 (expr=0, jaw=0)
        expr_params = torch.zeros(1, 50, device=device)  # 中性表情
        jaw_params = torch.zeros(1, 1, device=device)    # 中性jaw姿态

        # 生成基准顶点
        with torch.no_grad():
            vertices = flame_model(expr_params, jaw_params, shape=shape_params)
            # vertices shape: (1, 5023, 3)

        # 转换为numpy并reshape为 (5023*3,)
        template_verts = vertices.squeeze(0).cpu().numpy()  # (5023, 3)
        templates[subject_id] = template_verts

        print(f"  生成顶点形状: {template_verts.shape}")

    # 保存到pickle文件
    with open(output_path, 'wb') as f:
        pickle.dump(templates, f)

    print(f"模板已保存到: {output_path}")
    print(f"模板包含 {len(templates)} 个subjects")

    return templates

def create_vocaset_flame_template(output_path='vocaset/templates_flame.pkl'):
    """为vocaset数据集创建FLAME模板"""
    # vocaset的subjects列表（基于配置文件）
    subjects_list = [
        'FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170725_00137_TA',
        'FaceTalk_170915_00223_TA', 'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA',
        'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170811_03275_TA',
        'FaceTalk_170908_03277_TA', 'FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA'
    ]

    return create_flame_template(subjects_list, output_path)

def create_biwi_flame_template(output_path='BIWI/templates_flame.pkl'):
    """为BIWI数据集创建FLAME模板"""
    # BIWI的subjects列表
    subjects_list = ['F1','F2','F3','F4','F5','F6','F7','F8','M1','M2','M3','M4','M5','M6']

    return create_flame_template(subjects_list, output_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成FLAME模板文件')
    parser.add_argument('--dataset', type=str, choices=['vocaset', 'biwi', 'custom'],
                       default='vocaset', help='数据集类型')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--subjects', type=str, nargs='+', help='自定义subjects列表')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')

    args = parser.parse_args()

    if args.dataset == 'vocaset':
        output_path = args.output or 'vocaset/templates_flame.pkl'
        create_vocaset_flame_template(output_path)
    elif args.dataset == 'biwi':
        output_path = args.output or 'BIWI/templates_flame.pkl'
        create_biwi_flame_template(output_path)
    elif args.dataset == 'custom':
        if not args.subjects:
            raise ValueError("使用--dataset custom时必须提供--subjects参数")
        output_path = args.output or 'templates_custom.pkl'
        create_flame_template(args.subjects, output_path, args.device)
    else:
        print("请指定数据集类型: --dataset vocaset 或 --dataset biwi 或 --dataset custom")
