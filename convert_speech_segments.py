#!/usr/bin/env python3
"""
Convert result.json to continuous speech segments format
Extracts continuous speaking segments from track frames where is_speaking=True
"""

import json
import argparse
import os
from typing import List, Dict, Any

# 新增：最小平均得分阈值
MIN_SCORE_MEAN = 0.5

def extract_continuous_segments(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract continuous speaking segments from a track's frames
    """
    segments = []
    current_segment = None
    
    for frame in frames:
        if frame['is_speaking']:
            if current_segment is None:
                current_segment = {
                    'start_timestamp': frame['timestamp'],
                    'start_frame': frame['frame_id'],
                    'scores': [frame['score']],
                    'frames': [frame]
                }
            else:
                current_segment['scores'].append(frame['score'])
                current_segment['frames'].append(frame)
        else:
            if current_segment is not None:
                current_segment['end_timestamp'] = current_segment['frames'][-1]['timestamp']
                current_segment['end_frame'] = current_segment['frames'][-1]['frame_id']
                current_segment['scores_mean'] = sum(current_segment['scores']) / len(current_segment['scores'])
                current_segment['duration'] = current_segment['end_timestamp'] - current_segment['start_timestamp']
                
                # 修改：同时判断持续时间与平均得分阈值
                if current_segment['duration'] > 0 and current_segment['scores_mean'] >= MIN_SCORE_MEAN:
                    segments.append({
                        'start': current_segment['start_timestamp'],
                        'end': current_segment['end_timestamp'],
                        'is_speaking': True,
                        'scores_mean': round(current_segment['scores_mean'], 3),
                        'duration': round(current_segment['duration'], 3),
                        'start_frame': current_segment['start_frame'],
                        'end_frame': current_segment['end_frame']
                    })
                current_segment = None
    
    # Handle segment ending at last frame
    if current_segment is not None:
        current_segment['end_timestamp'] = current_segment['frames'][-1]['timestamp']
        current_segment['end_frame'] = current_segment['frames'][-1]['frame_id']
        current_segment['scores_mean'] = sum(current_segment['scores']) / len(current_segment['scores'])
        current_segment['duration'] = current_segment['end_timestamp'] - current_segment['start_timestamp']
        
        if current_segment['duration'] > 0 and current_segment['scores_mean'] >= MIN_SCORE_MEAN:
            segments.append({
                'start': current_segment['start_timestamp'],
                'end': current_segment['end_timestamp'],
                'is_speaking': True,
                'scores_mean': round(current_segment['scores_mean'], 3),
                'duration': round(current_segment['duration'], 3),
                'start_frame': current_segment['start_frame'],
                'end_frame': current_segment['end_frame']
            })
    
    return segments


def convert_json(input_file: str, output_file: str) -> None:
    """
    Convert result.json to speech segments format
    
    Args:
        input_file: Path to input result.json
        output_file: Path to output speech_segments.json
    """
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        tracks_data = json.load(f)
    
    # Process each track
    all_segments = []
    
    for track in tracks_data:
        track_id = track['track_id']
        frames = track['frames']
        
        # Extract continuous speaking segments
        segments = extract_continuous_segments(frames)
        
        # Add track_id to each segment
        for segment in segments:
            segment['track_id'] = track_id
            all_segments.append(segment)
    
    # Sort segments by start time
    all_segments.sort(key=lambda x: (x['track_id'], x['start']))
    
    # Generate summary
    summary = {
        'total_segments': len(all_segments),
        'total_duration': round(sum(s['duration'] for s in all_segments), 3),
        'tracks_with_speech': len(set(s['track_id'] for s in all_segments))
    }
    
    # Prepare output
    output_data = {
        'summary': summary,
        'segments': all_segments
    }
    
    # Save output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总片段数: {summary['total_segments']}")
    print(f"总说话时长: {summary['total_duration']}秒")
    print(f"有说话的track数: {summary['tracks_with_speech']}")


def main():
    parser = argparse.ArgumentParser(description='Convert result.json to speech segments format')
    parser.add_argument('--input', '-i', type=str, default='demo/test/pywork/result.json',
                        help='Input result.json file path')
    parser.add_argument('--output', '-o', type=str, default='demo/test/speech_segments.json',
                        help='Output speech segments file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误：输入文件 {args.input} 不存在")
        return
    
    convert_json(args.input, args.output)


if __name__ == '__main__':
    main()