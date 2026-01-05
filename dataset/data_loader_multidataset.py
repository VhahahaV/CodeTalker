#!/usr/bin/env python3
"""
MultiDataset 数据加载器
用于加载用户的自定义数据集（MultiModal200, MEAD_VHAP, digitalhuman）
这些数据集使用51维motion数据，而不是顶点数据
"""

import os
import torch
import numpy as np
import pickle
import json
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data

from utils.flame_model import FLAMEModel

class MotionDataset(data.Dataset):
    """Motion数据集类，用于51维motion -> 顶点数据"""

    def __init__(self, data, subjects_dict, data_type="train", read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio

    def __getitem__(self, index):
        """返回一个数据对 (motion, template, one_hot, file_name)"""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]  # 顶点数据 (T, 15069)
        template = self.data[index]["template"]  # 顶点模板 (15069,)

        if self.data_type == "train":
            # 从speaker_id中提取speaker名称
            speaker_id = self.data[index]["speaker_id"]
            speaker_name = speaker_id[0]  # speaker_id是[speaker_name, emotion]格式
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(speaker_name)]
        else:
            one_hot = self.one_hot_labels

        if self.read_audio:
            return (
                torch.FloatTensor(audio) if audio is not None else torch.zeros(1),
                torch.FloatTensor(vertice),
                torch.FloatTensor(template),
                torch.FloatTensor(one_hot),
                file_name
            )
        else:
            return torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def load_json_data(json_path, data_root):
    """从JSON文件加载数据"""
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    data_list = []
    for key, item in json_data.items():
        data_item = {
            'name': key,
            'speaker_id': item['speaker_id'],
            'audio_path': os.path.join(data_root, item['audio_path']),
            'motion_path': os.path.join(data_root, item.get('motion_feature_path', item.get('flame_coeff_save_path', ''))),
            'split': item.get('split', 'train')
        }
        data_list.append(data_item)

    return data_list

def read_multidataset_data(args):
    """读取multidataset数据"""
    print("Loading multidataset data...")

    # 加载Wav2Vec2处理器（如果需要音频）
    processor = None
    if args.read_audio:
        try:
            processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2model_path)
        except:
            print("Warning: Could not load Wav2Vec2 processor")

    # 加载模板
    if hasattr(args, 'template_file'):
        # 如果指定了template_file，检查是否是绝对路径
        if os.path.isabs(args.template_file):
            template_file = args.template_file
        else:
            # 如果是相对路径，相对于项目根目录（而不是data_root）
            template_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.template_file)
    else:
        template_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'multidataset/templates.pkl')
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    # 收集所有数据
    all_train_data = []
    all_val_data = []
    all_test_data = []

    # 处理训练数据JSON
    if hasattr(args, 'data_jsons'):
        for json_file in args.data_jsons:
            json_path = os.path.join(args.data_root, json_file)
            if os.path.exists(json_path):
                json_data = load_json_data(json_path, args.data_root)
                for item in json_data:
                    if item['split'] == 'train':
                        all_train_data.append(item)

    # 处理验证数据JSON
    if hasattr(args, 'val_data_jsons'):
        for json_file in args.val_data_jsons:
            json_path = os.path.join(args.data_root, json_file)
            if os.path.exists(json_path):
                json_data = load_json_data(json_path, args.data_root)
                for item in json_data:
                    if item['split'] in ['val', 'valid']:
                        all_val_data.append(item)

    # 处理测试数据JSON
    if hasattr(args, 'test_data_jsons'):
        for json_file in args.test_data_jsons:
            json_path = os.path.join(args.data_root, json_file)
            if os.path.exists(json_path):
                json_data = load_json_data(json_path, args.data_root)
                for item in json_data:
                    if item['split'] == 'test':
                        all_test_data.append(item)

    # 初始化FLAME（用于将51维motion转换为顶点）
    flame_model = None
    try:
        flame_model = FLAMEModel(flame_model_path=getattr(args, 'flame_model_path', None), device='cpu')
        flame_model.eval()
    except Exception as e:
        print(f"Warning: 初始化FLAME失败，后续将使用零顶点占位: {e}")
        flame_model = None

    # 处理数据：加载音频、motion并转为顶点
    processed_train = []
    processed_val = []
    processed_test = []

    print(f"Processing {len(all_train_data)} training samples...")
    for item in tqdm(all_train_data):
        processed_item = process_multidataset_item(item, processor, templates, args, flame_model)
        if processed_item:
            processed_train.append(processed_item)

    print(f"Processing {len(all_val_data)} validation samples...")
    for item in tqdm(all_val_data):
        processed_item = process_multidataset_item(item, processor, templates, args, flame_model)
        if processed_item:
            processed_val.append(processed_item)

    print(f"Processing {len(all_test_data)} test samples...")
    for item in tqdm(all_test_data):
        processed_item = process_multidataset_item(item, processor, templates, args, flame_model)
        if processed_item:
            processed_test.append(processed_item)

    # 构建subjects字典
    subjects_dict = {"train": [], "val": [], "test": []}

    # 从训练数据中提取唯一的speakers
    train_speakers = set()
    for item in processed_train:
        speaker_name = item["speaker_id"][0]
        train_speakers.add(speaker_name)

    subjects_dict["train"] = sorted(list(train_speakers))
    subjects_dict["val"] = subjects_dict["train"]  # 验证和测试使用相同的speakers
    subjects_dict["test"] = subjects_dict["train"]

    print(f'Loaded multidataset: Train-{len(processed_train)}, Val-{len(processed_val)}, Test-{len(processed_test)}')
    print(f'Train speakers: {len(subjects_dict["train"])}')

    return processed_train, processed_val, processed_test, subjects_dict

def process_multidataset_item(item, processor, templates, args, flame_model):
    """处理单个数据项"""
    try:
        # 加载音频（如果需要）
        audio_data = None
        if args.read_audio and processor and os.path.exists(item['audio_path']):
            try:
                speech_array, sampling_rate = librosa.load(item['audio_path'], sr=16000)
                audio_data = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
            except Exception as e:
                print(f"Warning: Could not load audio {item['audio_path']}: {e}")
                return None

        # 加载motion数据
        motion_data = None
        if os.path.exists(item['motion_path']):
            try:
                if item['motion_path'].endswith('.npz'):
                    motion_npz = np.load(item['motion_path'], allow_pickle=True)
                    # 尝试不同的键名
                    for key in ['images', 'data', 'motion', 'features']:
                        if key in motion_npz:
                            motion_data = motion_npz[key]
                            break
                    if motion_data is None:
                        # 如果没有找到标准键，尝试第一个数组
                        arrays = [motion_npz[f] for f in motion_npz.files]
                        if arrays:
                            motion_data = arrays[0]
                elif item['motion_path'].endswith('.npy'):
                    motion_data = np.load(item['motion_path'], allow_pickle=True)
                else:
                    print(f"Warning: Unsupported motion file format: {item['motion_path']}")
                    return None
            except Exception as e:
                print(f"Warning: Could not load motion {item['motion_path']}: {e}")
                return None

        if motion_data is None:
            print(f"Warning: No motion data found for {item['name']}")
            return None

        # 确保motion数据是正确的形状 (T, 51)
        motion_data = np.array(motion_data)
        if motion_data.ndim == 1:
            motion_data = motion_data.reshape(1, -1)
        elif motion_data.ndim > 2:
            motion_data = motion_data.reshape(motion_data.shape[0], -1)

        # 使用FLAME将motion转换为顶点
        vertices_flat = None
        template_flat = None
        if flame_model is not None and flame_model.flame is not None:
            with torch.no_grad():
                motion_tensor = torch.from_numpy(motion_data).float().unsqueeze(0)  # (1, T, 51)
                vertices = flame_model.motion_to_vertices(motion_tensor)  # (1, T, 5023, 3)
                vertices = vertices.squeeze(0).cpu().numpy()  # (T, 5023, 3)
                vertices_flat = vertices.reshape(vertices.shape[0], -1)  # (T, 15069)
                template_flat = vertices_flat[0]
        else:
            # 如果FLAME不可用，使用零占位并继续
            T = motion_data.shape[0]
            vertices_flat = np.zeros((T, 5023 * 3), dtype=np.float32)
            template_flat = np.zeros(5023 * 3, dtype=np.float32)

        # 如果提供了templates.pkl，则可以覆盖模板为预生成的值
        speaker_name = item["speaker_id"][0]
        if templates and speaker_name in templates:
            tmpl = templates[speaker_name]
            # 模板可能是51维motion或顶点，尝试匹配长度
            if tmpl.size == 5023 * 3:
                template_flat = tmpl.reshape(-1).astype(np.float32)
            elif tmpl.size == 51:
                # 如果是motion模板，忽略，保持FLAME生成的模板
                pass

        # 构建返回数据
        processed_item = {
            "name": item["name"],
            "audio": audio_data,
            "vertice": vertices_flat,
            "template": template_flat,
            "speaker_id": item["speaker_id"]
        }

        return processed_item

    except Exception as e:
        print(f"Error processing item {item['name']}: {e}")
        return None

def get_multidataset_dataloaders(args):
    """获取multidataset的数据加载器"""
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_multidataset_data(args)

    # 将speaker列表写回args，供模型使用（style embedding等）
    args.train_subjects = ' '.join(subjects_dict["train"])

    train_data = MotionDataset(train_data, subjects_dict, "train", args.read_audio)
    dataset["train"] = data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(args.workers, 4),  # 限制workers数量避免内存问题
        drop_last=True
    )

    valid_data = MotionDataset(valid_data, subjects_dict, "val", args.read_audio)
    dataset["valid"] = data.DataLoader(
        dataset=valid_data,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.workers, 2)
    )

    test_data = MotionDataset(test_data, subjects_dict, "test", args.read_audio)
    dataset["test"] = data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.workers, 2)
    )

    return dataset
