#!/usr/bin/env python3
"""
测试脚本：验证说话人标签JSON生成功能
使用示例: python test_speaker_json.py
"""

import subprocess
import os
import json
import sys

def test_speaker_json_generation():
    """测试说话人JSON生成功能"""
    
    print("=" * 60)
    print("测试说话人标签JSON生成功能")
    print("=" * 60)
    
    # 检查demo视频是否存在
    demo_videos = [
        "demo/001.mp4",
        "demo/1.mp4", 
        "demo/102.mp4"
    ]
    
    test_video = None
    for video in demo_videos:
        if os.path.exists(video):
            test_video = video
            break
    
    if test_video is None:
        print("错误: 找不到demo视频文件")
        print("请确保以下文件之一存在:")
        for video in demo_videos:
            print(f"  - {video}")
        return False
    
    video_name = os.path.splitext(os.path.basename(test_video))[0]
    print(f"使用测试视频: {test_video}")
    print(f"视频名称: {video_name}")
    
    # 构建命令
    cmd = [
        "python", "Columbia_test.py",
        "--videoName", video_name,
        "--videoFolder", "demo",
        "--generateJson",
        "--jsonOutputPath", f"demo/{video_name}_speaker_labels.json",
        "--speakerThreshold", "0.75"
    ]
    
    print("\n执行命令:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)
    
    try:
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        # 输出结果
        if result.stdout:
            print("标准输出:")
            print(result.stdout)
        
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        
        print(f"\n返回码: {result.returncode}")
        
        # 检查JSON文件是否生成
        json_file = f"demo/{video_name}_speaker_labels.json"
        if os.path.exists(json_file):
            print(f"\n✅ JSON文件成功生成: {json_file}")
            
            # 验证JSON文件内容
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print("\n📊 JSON文件内容验证:")
                print(f"  - 视频帧数: {data['video_info']['total_frames']}")
                print(f"  - 视频时长: {data['video_info']['duration']}秒")
                print(f"  - 检测到的说话人: {data['video_info']['detected_speakers']}")
                print(f"  - 总track数: {data['video_info']['total_tracks']}")
                
                # 检查说话人统计
                speaker_mapping = data['video_info']['speaker_mapping']
                print(f"\n👥 说话人详细信息:")
                for speaker_id, info in speaker_mapping.items():
                    print(f"  - {speaker_id}: {info['total_tracks']} 个tracks")
                
                # 检查前几帧的数据
                print(f"\n🎬 前5帧数据示例:")
                for i in range(min(5, len(data['frame_speakers']))):
                    frame = data['frame_speakers'][i]
                    print(f"  帧 {frame['frame_id']} ({frame['timestamp']}s): {len(frame['active_speakers'])} 个活跃说话人")
                    for speaker in frame['active_speakers']:
                        status = "🗣️ 说话" if speaker['is_speaking'] else "🤐 静默"
                        print(f"    - {speaker['speaker_id']}: {status} (置信度: {speaker['confidence']:.2f})")
                
                print(f"\n✅ JSON文件验证通过!")
                return True
                
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON文件格式错误: {e}")
                return False
                
        else:
            print(f"\n❌ JSON文件未生成: {json_file}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n❌ 命令执行超时")
        return False
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        return False

def show_usage():
    """显示使用方法"""
    print("\n🚀 使用方法:")
    print("1. 基本用法 - 生成说话人标签JSON:")
    print("   python Columbia_test.py --videoName YOUR_VIDEO --videoFolder demo --generateJson")
    print("\n2. 自定义输出路径:")
    print("   python Columbia_test.py --videoName YOUR_VIDEO --videoFolder demo --generateJson --jsonOutputPath output.json")
    print("\n3. 调整说话人聚类阈值:")
    print("   python Columbia_test.py --videoName YOUR_VIDEO --videoFolder demo --generateJson --speakerThreshold 0.8")
    print("\n4. 生成的JSON格式:")
    print("""   {
     "video_info": {
       "fps": 25,
       "total_frames": 500,
       "duration": 20.0,
       "detected_speakers": ["Speaker_1", "Speaker_2"],
       "speaker_mapping": {...}
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
   }""")

if __name__ == "__main__":
    success = test_speaker_json_generation()
    show_usage()
    
    if success:
        print("\n🎉 测试完成! 说话人标签JSON功能工作正常。")
        sys.exit(0)
    else:
        print("\n❌ 测试失败! 请检查错误信息并修复问题。")
        sys.exit(1) 