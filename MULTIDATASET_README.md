# MultiDataset Template 生成工具

## 核心文件

- **`create_multidataset_template.py`**: 生成multidataset模板的主要工具
- **`multidataset/templates.pkl`**: 生成的模板文件（109个speaker的51维motion模板）
- **`dataset/data_loader_multidataset.py`**: multidataset专用数据加载器
- **`config/multidataset/stage1.yaml`**: 已更新的配置文件
- **`models/__init__.py`**: 已更新的模型初始化文件（支持stage1_motion）
- **`test_multidataset_setup.py`**: 设置验证脚本

## 快速使用

### 生成模板（如果需要重新生成）
```bash
python create_multidataset_template.py --data_root /home/caizhuoqiang/Data
```

### 验证设置
```bash
python test_multidataset_setup.py
```

### 开始训练
```bash
python main/train_vq.py --config config/multidataset/stage1.yaml
```

## 数据集信息

- **MultiModal200**: 52个speakers
- **MEAD_VHAP**: 23个speakers
- **digital_human**: 15个speakers
- **总计**: 109个独特的speakers

## 技术规格

- **数据类型**: 51维motion数据（50维表情 + 1维下巴）
- **模板格式**: 每个speaker的51维虚拟基准参数
- **用途**: CodeTalker相对偏移学习

## 注意事项

模板文件已生成并配置，无需额外操作即可开始训练。
