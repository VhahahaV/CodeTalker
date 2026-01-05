#!/usr/bin/env python
"""
推理脚本
根据baseline_prompt.md要求，实现多数据集推理
输出格式：results/metrics/<DATASET>/<STYLE_ID>/0.npy
"""
import os
import json
import numpy as np
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Processor

from base.utilities import get_parser
from models import get_model
from base.baseTrainer import load_state_dict
from dataset.data_item import DataItem


def build_style_id(speaker_id, emotion):
    """
    构建STYLE_ID：["speaker_id", "emotion"]_passionate
    
    Args:
        speaker_id: speaker ID字符串
        emotion: emotion字符串
    
    Returns:
        style_id: 格式化的STYLE_ID字符串
    """
    return f'["{speaker_id}", "{emotion}"]_passionate'


def detect_dataset_from_path(flame_path):
    """从路径检测数据集名称"""
    flame_path_lower = flame_path.lower()
    if 'multimodal' in flame_path_lower or 'multimodal200' in flame_path_lower:
        return 'MultiModal200'
    elif 'mead' in flame_path_lower or 'vhap' in flame_path_lower:
        return 'MEAD_VHAP'
    else:
        return 'MEAD_VHAP'  # 默认


def main():
    args = get_parser()
    
    # 设置设备
    device = args.device if hasattr(args, 'device') else 'cuda'
    torch.cuda.set_device(0)
    
    # 加载模型
    print("Loading model...")
    model = get_model(args)
    
    # 加载checkpoint
    if hasattr(args, 'model_path') and args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'state_dict' in checkpoint:
            load_state_dict(model, checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.model_path}")
    
    model.eval()
    model = model.cuda()
    
    # 初始化Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2model_path)
    
    # 解析data_root和data_jsons
    # 根据baseline_prompt.md，data_root统一为/home/caizhuoqiang/Data
    if isinstance(args.data_root, list):
        unified_data_root = args.data_root[0]  # 使用第一个（应该都是同一个）
    else:
        unified_data_root = args.data_root
    
    test_jsons = args.test_data_jsons if isinstance(args.test_data_jsons, list) else [args.test_data_jsons]
    
    # 创建输出目录
    output_base = 'results/metrics'
    os.makedirs(output_base, exist_ok=True)
    
    # 统计信息
    metrics_report = {}
    
    # 遍历所有测试JSON文件
    # 所有JSON路径都是相对于统一的data_root
    for test_json in test_jsons:
        json_path = os.path.join(unified_data_root, test_json)
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found: {json_path}")
            continue
        
        # 加载JSON文件
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        
        # 检测数据集名称
        dataset_name = None
        for sample_id, sample_info in list(data_dict.items())[:1]:
            flame_path = sample_info.get('flame_coeff_save_path') or sample_info.get('flame_path', '')
            dataset_name = detect_dataset_from_path(flame_path)
            break
        
        if dataset_name is None:
            dataset_name = 'MEAD_VHAP'  # 默认
        
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Total samples: {len(data_dict)}")
        
        # 创建数据集输出目录
        dataset_output_dir = os.path.join(output_base, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 处理每个样本
        for sample_id, sample_info in tqdm(data_dict.items(), desc=f"Processing {dataset_name}"):
            try:
                # 创建DataItem，使用统一的data_root
                data_item = DataItem(sample_id, sample_info, unified_data_root, flame_model=None)
                
                # 获取speaker_id和emotion
                speaker_id = sample_info.get('speaker_id', '')
                emotion = sample_info.get('emotion', '')
                
                # 构建STYLE_ID
                style_id = build_style_id(speaker_id, emotion)
                
                # 创建STYLE_ID输出目录
                style_output_dir = os.path.join(dataset_output_dir, style_id)
                os.makedirs(style_output_dir, exist_ok=True)
                
                # 加载音频
                audio_path = data_item.get_audio_path()
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue
                
                speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
                audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                audio_feature = torch.FloatTensor(audio_feature).unsqueeze(0).cuda()  # (1, audio_len)
                
                # 获取template（中性表情顶点）
                # 这里我们需要一个模板，可以使用第一帧或零模板
                # 为了兼容性，我们使用零模板
                template = torch.zeros(5023 * 3).cuda()  # (5023*3,)
                
                # 获取speaker one-hot编码
                # 需要从训练数据中获取speaker列表
                if hasattr(args, 'train_subjects') and args.train_subjects:
                    speakers = args.train_subjects.split()
                else:
                    # 从数据中推断
                    speakers = set()
                    for sid, sinfo in data_dict.items():
                        if sinfo.get('speaker_id'):
                            speakers.add(sinfo['speaker_id'])
                    speakers = sorted(list(speakers))
                
                # 创建one-hot编码
                if speaker_id in speakers:
                    speaker_idx = speakers.index(speaker_id)
                    one_hot = torch.zeros(len(speakers))
                    one_hot[speaker_idx] = 1.0
                else:
                    # 如果speaker不在训练集中，使用第一个speaker
                    one_hot = torch.zeros(len(speakers))
                    if len(speakers) > 0:
                        one_hot[0] = 1.0
                
                one_hot = one_hot.unsqueeze(0).cuda()  # (1, n_speakers)
                
                # 推理
                with torch.no_grad():
                    vertices_pred = model.predict(audio_feature, template.unsqueeze(0), one_hot)
                    vertices_pred = vertices_pred.squeeze(0).cpu().numpy()  # (T, 5023*3)
                
                # 保存顶点文件
                output_path = os.path.join(style_output_dir, '0.npy')
                np.save(output_path, vertices_pred)
                
                # 更新统计信息
                if style_id not in metrics_report:
                    metrics_report[style_id] = {
                        'dataset': dataset_name,
                        'samples': []
                    }
                metrics_report[style_id]['samples'].append(sample_id)
                
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存metrics_report.json
        metrics_report_path = os.path.join(dataset_output_dir, 'metrics_report.json')
        with open(metrics_report_path, 'w') as f:
            json.dump(metrics_report, f, indent=2)
        print(f"Saved metrics report to {metrics_report_path}")
    
    print("\nInference completed!")
    print(f"Output directory: {output_base}")


if __name__ == '__main__':
    main()

