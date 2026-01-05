#!/usr/bin/env python3
"""
为用户的自定义数据集（MultiModal200, MEAD_VHAP, digitalhuman）创建虚拟template
这些数据集使用51维motion数据，template仅用于代码兼容性
"""

import os
import json
import pickle
import numpy as np
import argparse

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

def create_motion_template(speakers_list, output_path='multidataset/templates.pkl'):
    """
    为motion-based数据集创建虚拟template

    Args:
        speakers_list: speaker ID列表
        output_path: 输出文件路径
    """

    templates = {}

    print(f"为 {len(speakers_list)} 个speakers创建虚拟motion模板...")

    for speaker_id in speakers_list:
        print(f"处理 speaker: {speaker_id}")

        # 为motion数据集创建51维的虚拟模板（50维expr + 1维jaw）
        # 使用固定的随机种子确保可重现性
        np.random.seed(hash(speaker_id) % 2**32)

        # 创建一个51维的虚拟基准motion参数（中性表情）
        # expr (50维): 轻微的随机值，模拟个性化的中性表情基准
        expr_template = np.random.normal(0, 0.01, 50).astype(np.float32)

        # jaw (1维): 轻微的随机值，模拟个性化的中性jaw姿态基准
        jaw_template = np.random.normal(0, 0.005, 1).astype(np.float32)

        # 合并为51维motion template
        motion_template = np.concatenate([expr_template, jaw_template])

        templates[speaker_id] = motion_template

        print(f"  模板形状: {motion_template.shape}")
        print(f"  数据范围: [{motion_template.min():.6f}, {motion_template.max():.6f}]")

    # 保存到pickle文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(templates, f)

    print(f"\n虚拟motion模板已保存到: {output_path}")
    print(f"模板包含 {len(templates)} 个speakers")

    return templates

def create_multidataset_template(data_root='/home/caizhuoqiang/Data', output_path='multidataset/templates.pkl'):
    """
    为multidataset创建template，包含所有三个数据集的speakers

    Args:
        data_root: 数据根目录
        output_path: 输出路径
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

    # 收集所有独特的speakers
    all_speakers = set()

    print("扫描所有数据集JSON文件以提取speakers...")
    for json_file in json_files:
        json_path = os.path.join(data_root, json_file)
        if os.path.exists(json_path):
            speakers = extract_unique_speakers_from_json(json_path)
            all_speakers.update(speakers)
            print(f"  {json_file}: 找到 {len(speakers)} 个speakers")
        else:
            print(f"  警告: {json_path} 不存在，跳过")

    # 转换为排序列表以确保一致性
    speakers_list = sorted(list(all_speakers))
    print(f"\n总共找到 {len(speakers_list)} 个独特的speakers")
    print("前10个speakers:", speakers_list[:10])

    # 创建template
    return create_motion_template(speakers_list, output_path)

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
    parser = argparse.ArgumentParser(description='为用户自定义数据集创建虚拟motion模板')
    parser.add_argument('--data_root', type=str, default='/home/caizhuoqiang/Data',
                       help='数据根目录')
    parser.add_argument('--output', type=str, default='multidataset/templates.pkl',
                       help='输出文件路径')
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
        create_multidataset_template(args.data_root, args.output)
