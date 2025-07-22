# 说话人标签JSON生成功能

## 功能概述

本功能为LR-ASD模型新增了**说话人标签JSON生成**能力，可以为视频的每一帧生成详细的说话人信息，包括说话人身份、是否在说话、置信度和人脸位置等。这个JSON文件可以直接用于视频翻译等下游任务。

## 核心特性

- ✅ **人脸特征提取**: 利用模型的128维视觉特征向量
- ✅ **说话人聚类**: 基于余弦相似度自动识别同一说话人
- ✅ **帧级别标签**: 为每一帧生成详细的说话人信息
- ✅ **平滑处理**: 对说话检测分数进行时间平滑
- ✅ **可配置阈值**: 支持调整说话人聚类的相似度阈值

## 使用方法

### 1. 基本使用

```bash
python Columbia_test.py --videoName YOUR_VIDEO --videoFolder demo --generateJson
```

### 2. 自定义输出路径

```bash
python Columbia_test.py --videoName YOUR_VIDEO --videoFolder demo --generateJson --jsonOutputPath my_output.json
```

### 3. 调整说话人聚类阈值

```bash
python Columbia_test.py --videoName YOUR_VIDEO --videoFolder demo --generateJson --speakerThreshold 0.8
```

## 新增参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--generateJson` | flag | False | 启用说话人标签JSON生成功能 |
| `--jsonOutputPath` | str | "speaker_labels.json" | JSON文件输出路径 |
| `--speakerThreshold` | float | 0.75 | 说话人聚类的相似度阈值 (0-1) |

## 输出JSON格式

```json
{
  "video_info": {
    "fps": 25,
    "total_frames": 500,
    "duration": 20.0,
    "detected_speakers": ["Speaker_1", "Speaker_2"],
    "total_tracks": 5,
    "speaker_mapping": {
      "Speaker_1": {
        "tracks": [0, 2],
        "total_tracks": 2
      },
      "Speaker_2": {
        "tracks": [1, 3, 4],
        "total_tracks": 3
      }
    }
  },
  "frame_speakers": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "active_speakers": [
                 {
           "speaker_id": "Speaker_1",
           "is_speaking": true,
           "confidence": 0.85
         }
      ]
    }
  ]
}
```

### JSON字段说明

#### video_info 部分
- `fps`: 视频帧率 (固定为25)
- `total_frames`: 视频总帧数
- `duration`: 视频总时长 (秒)
- `detected_speakers`: 检测到的所有说话人ID列表
- `total_tracks`: 总人脸追踪数量
- `speaker_mapping`: 说话人与tracks的映射关系

#### frame_speakers 部分
- `frame_id`: 帧编号 (从0开始)
- `timestamp`: 时间戳 (秒)
- `active_speakers`: 该帧中的所有活跃说话人

#### active_speakers 部分
- `speaker_id`: 说话人ID (如 "Speaker_1")
- `is_speaking`: 是否在说话 (boolean)
- `confidence`: 说话检测置信度 (float, >0为说话)

## 说话人识别原理

### 1. 人脸特征提取
利用LR-ASD模型的视觉编码器提取每个track的128维人脸特征向量。

### 2. 特征聚类
使用余弦相似度计算特征向量之间的相似性，当相似度超过阈值时，认为属于同一说话人。

### 3. 说话人ID分配
为每个聚类分配唯一的说话人ID（Speaker_1, Speaker_2等）。

### 4. 帧级别映射
将track级别的信息映射到每一帧，生成完整的时间序列标签。

## 参数调优建议

### speakerThreshold 阈值选择

| 阈值范围 | 效果 | 适用场景 |
|----------|------|----------|
| 0.6-0.7 | 说话人数量较少，容易合并 | 人脸变化较大的视频 |
| 0.75-0.8 | 平衡的聚类效果 | 大多数普通视频 |
| 0.85-0.9 | 说话人数量较多，更严格 | 人脸变化较小的视频 |

## 使用示例

### 示例1: 处理demo视频

```bash
# 处理demo目录下的001.mp4
python Columbia_test.py --videoName 001 --videoFolder demo --generateJson

# 输出文件: demo/001/speaker_labels.json
```

### 示例2: 批处理多个视频

```bash
#!/bin/bash
for video in demo/*.mp4; do
    name=$(basename "$video" .mp4)
    python Columbia_test.py --videoName "$name" --videoFolder demo --generateJson --jsonOutputPath "output/${name}_speakers.json"
done
```

## 测试验证

使用提供的测试脚本验证功能：

```bash
python test_speaker_json.py
```

测试脚本会：
1. 自动查找demo视频
2. 执行说话人标签生成
3. 验证JSON文件格式和内容
4. 显示详细的结果统计

## 应用场景

### 1. 视频翻译
```python
import json

# 加载说话人标签
with open('speaker_labels.json', 'r') as f:
    data = json.load(f)

# 为每个说话人分配不同的翻译配置
for frame in data['frame_speakers']:
    for speaker in frame['active_speakers']:
        if speaker['is_speaking']:
            # 根据speaker_id应用不同的翻译策略
            translate_segment(speaker['speaker_id'], frame['timestamp'])
```

### 2. 会议记录
根据说话人ID生成结构化的会议记录。

### 3. 视频分析
统计每个说话人的发言时间和频率。

## 注意事项

1. **GPU要求**: 需要CUDA兼容的GPU进行特征提取
2. **处理时间**: 启用特征提取会增加约20-30%的处理时间
3. **内存使用**: 特征存储会增加内存使用量
4. **阈值调试**: 建议根据具体视频调整speakerThreshold参数

## 故障排查

### 常见问题

1. **ImportError: sklearn not found**
   ```bash
   pip install scikit-learn
   ```

2. **JSON文件未生成**
   - 检查是否添加了 `--generateJson` 参数
   - 确认输出目录有写入权限

3. **说话人数量不正确**
   - 调整 `--speakerThreshold` 参数
   - 检查视频质量和人脸清晰度

## 更新日志

- **v1.0**: 初始版本，支持基本的说话人标签JSON生成
- 基于人脸特征的说话人聚类
- 帧级别的说话检测和标签生成
- 可配置的聚类阈值参数 