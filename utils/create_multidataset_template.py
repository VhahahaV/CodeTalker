#!/usr/bin/env python3
"""
为自定义多数据集（MultiModal200, MEAD_VHAP, digitalhuman）生成顶点模板
策略：
1) expression/jaw 全零（中性表情）
2) shape 使用该 speaker 的 shape 系数（从对应的 flame/motion 文件读取），无则用零 shape
3) 通过 FLAME forward 得到中性顶点 (5023x3)，保存为 templates.pkl
"""

import os
import json
import pickle
import numpy as np
import argparse
import torch

from utils.flame_model import FLAMEModel

def extract_unique_speakers_from_json(json_path):
    """
    从JSON文件中提取所有独特的speaker IDs

    Args:
        json_path: JSON文件路径

    Returns:
        set: 独特的speaker ID集合
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    speakers = set()
    for item_key, item_data in data.items():
        if 'speaker_id' in item_data:
            # speaker_id 是 [speaker_name, emotion] 格式，取第一个元素
            speaker_name = item_data['speaker_id'][0]
            speakers.add(speaker_name)

    return speakers

def load_shape_from_motion(motion_path):
    """尝试从 motion/flame 系数文件中读取 shape 参数"""
    if not os.path.exists(motion_path):
        return None
    try:
        if motion_path.endswith('.npz'):
            data = np.load(motion_path, allow_pickle=True)
            for key in ['shape', 'shapecode']:
                if key in data:
                    shape = data[key]
                    if shape.ndim > 1:
                        shape = shape[0]
                    return np.array(shape, dtype=np.float32)
        elif motion_path.endswith('.npy'):
            # npy 通常只有51维，不含shape
            return None
    except Exception as e:
        print(f"  Warning: 读取shape失败 {motion_path}: {e}")
    return None


def build_vertex_template(shape_params, flame_model):
    """使用 FLAME 将 expression/jaw 全零 + shape 转为中性顶点"""
    B, T = 1, 1
    expr = torch.zeros((B, T, 50), dtype=torch.float32)
    jaw = torch.zeros((B, T, 1), dtype=torch.float32)
    shape_t = None
    if shape_params is not None:
        shape_t = torch.from_numpy(shape_params.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        verts = flame_model(expr, jaw, shape=shape_t)  # (1,1,5023,3) 或零占位
        verts = verts.squeeze(0).squeeze(0).cpu().numpy()
    return verts


def create_vertex_templates(speakers_to_motion, flame_model_path, output_path='multidataset/templates.pkl'):
    """
    为每个 speaker 生成顶点模板（5023x3），并保存为 templates.pkl
    """
    templates = {}
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    flame = FLAMEModel(flame_model_path=flame_model_path, device='cpu')
    if flame.flame is None:
        print("Warning: FLAME 未正确加载，生成的模板将为全零。请安装 flame_pytorch 并检查模型路径。")

    for speaker, motion_path in speakers_to_motion.items():
        print(f"处理 speaker: {speaker}")
        shape_params = load_shape_from_motion(motion_path)
        verts = build_vertex_template(shape_params, flame)
        templates[speaker] = verts.astype(np.float32)
        print(f"  模板形状: {verts.shape}, min={verts.min():.6f}, max={verts.max():.6f}")

    with open(output_path, 'wb') as f:
        pickle.dump(templates, f)

    print(f"\n顶点模板已保存到: {output_path}")
    print(f"模板包含 {len(templates)} 个speakers")
    return templates

def create_multidataset_template(data_root='/home/caizhuoqiang/Data',
                                output_path='multidataset/templates.pkl',
                                flame_model_path='utils/FLAME2020/generic_model.pkl'):
    """
    为multidataset创建顶点模板，包含所有三个数据集的speakers

    Args:
        data_root: 数据根目录
        output_path: 输出路径
        flame_model_path: FLAME模型路径
    """

    # 定义JSON文件路径
    json_files = [
        'dataset_jsons/splits/MultiModal200_train.json',
        'dataset_jsons/splits/MultiModal200_val.json',
        'dataset_jsons/splits/MultiModal200_test.json',
        'dataset_jsons/splits/MEAD_VHAP_train.json',
        'dataset_jsons/splits/MEAD_VHAP_val.json',
        'dataset_jsons/splits/MEAD_VHAP_test.json',
        'dataset_jsons/splits/digital_human.json'
    ]

    # 收集每个 speaker 对应的一个 motion 路径（用于读取 shape）
    speakers_to_motion = {}

    print("扫描所有数据集JSON文件以提取speakers和motion路径...")
    for json_file in json_files:
        json_path = os.path.join(data_root, json_file)
        if not os.path.exists(json_path):
            print(f"  警告: {json_path} 不存在，跳过")
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        for _, item in data.items():
            if 'speaker_id' not in item:
                continue
            speaker = item['speaker_id'][0]
            motion_rel = item.get('flame_coeff_save_path') or item.get('motion_feature_path')
            if not motion_rel:
                continue
            motion_path = os.path.join(data_root, motion_rel)
            if speaker not in speakers_to_motion and os.path.exists(motion_path):
                speakers_to_motion[speaker] = motion_path

    print(f"\n总共找到 {len(speakers_to_motion)} 个有 motion 的speakers")
    return create_vertex_templates(speakers_to_motion, flame_model_path, output_path)

def create_separate_templates(data_root='/home/caizhuoqiang/Data'):
    """
    为每个数据集分别创建template
    """

    datasets = {
        'MultiModal200': [
            'dataset_jsons/splits/MultiModal200_train.json',
            'dataset_jsons/splits/MultiModal200_val.json',
            'dataset_jsons/splits/MultiModal200_test.json'
        ],
        'MEAD_VHAP': [
            'dataset_jsons/splits/MEAD_VHAP_train.json',
            'dataset_jsons/splits/MEAD_VHAP_val.json',
            'dataset_jsons/splits/MEAD_VHAP_test.json'
        ],
        'digital_human': ['dataset_jsons/splits/digital_human.json']
    }

    for dataset_name, json_files in datasets.items():
        print(f"\n=== 处理 {dataset_name} 数据集 ===")

        # 收集speakers
        all_speakers = set()
        for json_file in json_files:
            json_path = os.path.join(data_root, json_file)
            if os.path.exists(json_path):
                speakers = extract_unique_speakers_from_json(json_path)
                all_speakers.update(speakers)

        speakers_list = sorted(list(all_speakers))
        output_path = f'{dataset_name}/templates.pkl'

        if speakers_list:
            create_motion_template(speakers_list, output_path)
        else:
            print(f"  {dataset_name}: 没有找到speakers")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为用户自定义数据集创建顶点模板')
    parser.add_argument('--data_root', type=str, default='/home/caizhuoqiang/Data',
                       help='数据根目录')
    parser.add_argument('--output', type=str, default='multidataset/templates.pkl',
                       help='输出文件路径')
    parser.add_argument('--flame_model', type=str, default='utils/FLAME2020/generic_model.pkl',
                       help='FLAME模型路径')
    parser.add_argument('--separate', action='store_true',
                       help='为每个数据集分别创建template')
    parser.add_argument('--dataset', type=str, choices=['multidataset', 'MultiModal200', 'MEAD_VHAP', 'digital_human'],
                       help='指定单个数据集')

    args = parser.parse_args()

    if args.separate:
        create_separate_templates(args.data_root)
    elif args.dataset:
        if args.dataset == 'multidataset':
            create_multidataset_template(args.data_root, args.output)
        else:
            # 处理单个数据集
            json_files_map = {
                'MultiModal200': [
                    'dataset_jsons/splits/MultiModal200_train.json',
                    'dataset_jsons/splits/MultiModal200_val.json',
                    'dataset_jsons/splits/MultiModal200_test.json'
                ],
                'MEAD_VHAP': [
                    'dataset_jsons/splits/MEAD_VHAP_train.json',
                    'dataset_jsons/splits/MEAD_VHAP_val.json',
                    'dataset_jsons/splits/MEAD_VHAP_test.json'
                ],
                'digital_human': ['dataset_jsons/splits/digital_human.json']
            }

            all_speakers = set()
            for json_file in json_files_map[args.dataset]:
                json_path = os.path.join(args.data_root, json_file)
                if os.path.exists(json_path):
                    speakers = extract_unique_speakers_from_json(json_path)
                    all_speakers.update(speakers)

            speakers_list = sorted(list(all_speakers))
            output_path = f'{args.dataset}/templates.pkl'

            if speakers_list:
                create_motion_template(speakers_list, output_path)
            else:
                print(f"没有找到 {args.dataset} 数据集的speakers")
    else:
        # 默认创建multidataset的统一template
        create_multidataset_template(args.data_root, args.output, args.flame_model)
