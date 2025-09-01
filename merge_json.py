import json
import argparse
from pathlib import Path

def merge_sentences(sentences_to_merge):
    """
    Merges a list of sentence objects into a single sentence object.
    """
    if not sentences_to_merge:
        return None

    # Use the first sentence as the base for the merged sentence
    merged = sentences_to_merge[0].copy()

    # Concatenate text and translated_text
    merged['text'] = " ".join(s.get('text', '') for s in sentences_to_merge).strip()
    merged['translated_text'] = " ".join(s.get('translated_text', '') for s in sentences_to_merge).strip()

    # Set start to the minimum start time and end to the maximum end time
    merged['start'] = min(s['start'] for s in sentences_to_merge)
    merged['end'] = max(s['end'] for s in sentences_to_merge)

    # Update duration
    merged['duration'] = merged['end'] - merged['start']

    # Combine speaker IDs, ensuring uniqueness and order
    speaker_ids = []
    for s in sentences_to_merge:
        speaker_id = s.get('speaker')
        if speaker_id and speaker_id not in speaker_ids:
            speaker_ids.append(speaker_id)
    merged['speaker'] = ", ".join(speaker_ids)
    
    # Note: Other fields like embeddings, audio paths, etc., are taken from the first sentence.
    # This behavior can be adjusted if different logic is needed.

    return merged

def process_json_files(speech_segments_path, original_json_path, output_json_path):
    """
    Processes the JSON files to merge sentence segments based on speech segments.
    """
    try:
        with open(speech_segments_path, 'r', encoding='utf-8') as f:
            speech_data = json.load(f)
        
        with open(original_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from a file: {e}")
        return

    original_sentences = original_data.get('sentence_list', [])
    speech_segments = speech_data.get('segments', [])

    new_sentence_list = []
    processed_indices = set()

    for segment in speech_segments:
        segment_start_ms = segment['start'] * 1000
        segment_end_ms = segment['end'] * 1000

        sentences_in_segment = []
        indices_in_segment = []

        for i, sentence in enumerate(original_sentences):
            # Check for overlap
            if sentence['start'] < segment_end_ms and sentence['end'] > segment_start_ms:
                sentences_in_segment.append(sentence)
                indices_in_segment.append(i)

        if len(sentences_in_segment) > 1:
            # More than one sentence falls into the segment, so merge them.
            print(f"Merging {len(sentences_in_segment)} sentences for speech segment {segment['start']}-{segment['end']}")
            merged_sentence = merge_sentences(sentences_in_segment)
            if merged_sentence:
                new_sentence_list.append(merged_sentence)
                processed_indices.update(indices_in_segment)
        elif len(sentences_in_segment) == 1:
            # Only one sentence, check if it's already processed
            if indices_in_segment[0] not in processed_indices:
                new_sentence_list.append(sentences_in_segment[0])
                processed_indices.add(indices_in_segment[0])


    # Add any remaining sentences that were not part of any merge operation
    for i, sentence in enumerate(original_sentences):
        if i not in processed_indices:
            new_sentence_list.append(sentence)
            
    # Sort the final list by start time
    new_sentence_list.sort(key=lambda x: x['start'])

    # Create the new JSON data
    output_data = original_data.copy()
    output_data['sentence_list'] = new_sentence_list

    # Write the output file
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Merged JSON saved to: {output_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge sentence segments in a JSON file based on speech segments.')
    parser.add_argument('--speech_segments', type=str, required=True, help='Path to the speech segments JSON file (e.g., speech_segments.json).')
    parser.add_argument('--original_json', type=str, required=True, help='Path to the original JSON file to be modified (e.g., test_3.json).')
    parser.add_argument('--output_json', type=str, required=True, help='Path to save the modified JSON file.')

    args = parser.parse_args()

    process_json_files(args.speech_segments, args.original_json, args.output_json)
