# motionepic/dataset/star_dataset.py
import os
import pickle
import bisect
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import av
import json
from motionepic.constants import DEFAULT_VIDEO_TOKEN, DEFAULT_SG_TOKEN

class STARDataset(Dataset):
    """Dataset for supervised fine-tuning of MotionEpic on the STAR dataset."""

    def __init__(self, data_path, tokenizer, data_args, video_folder=None):
        super(STARDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.video_folder = video_folder
        
        # Load the dataset
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Process data into the format needed by MotionEpic
        self.list_data_dict = self._prepare_conversations()
    
    def _prepare_conversations(self):
        list_data_dict = []
        for item in self.data:
            video_id = item["video_id"]
            question = item["question"]
            choices = [choice["choice"] for choice in item["choices"]]
            answer = item["answer"]
            
            # Construct the input conversation
            user_prompt = f"{DEFAULT_VIDEO_TOKEN}\nGiven the question: {question}\nOptions: {' | '.join(choices)}\nPlease select the correct answer."
            
            # Construct the expected assistant response
            assistant_response = f"The correct answer is: {answer}"
            
            # Create the conversation dictionary in the expected format
            list_item = {
                "input_video": os.path.join(self.video_folder, f"{video_id}.mp4"),
                "start": item["start"],
                "end": item["end"],
                "situations": item.get("situations", {}),  # STSG data
                "conversations": [
                    {"from": "human", "value": user_prompt},
                    {"from": "gpt", "value": assistant_response}
                ]
            }
            
            list_data_dict.append(list_item)
        
        return list_data_dict
    
    def __len__(self):
        return len(self.list_data_dict)
    
    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'input_video' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'input_video' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        
        if isinstance(i, int):
            sources = [sources]
            
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        
        if 'input_video' in sources[0]:
            video_file = self.list_data_dict[i]['input_video']
            video_folder = self.data_args.video_folder or self.video_folder
            processor = self.data_args.video_processor
            
            # Handle timestamps if available
            start_time = self.list_data_dict[i].get('start', None)
            end_time = self.list_data_dict[i].get('end', None)
            
            # Read video with timestamps if available
            if start_time is not None and end_time is not None:
                video_frames = self._read_video_with_timestamps(video_file, start_time, end_time)
            else:
                video_frames = self._read_video(video_file)
            
            # Process video frames
            if len(video_frames) > 0:
                _temp_frames = [processor(frame, return_tensors='pt')['pixel_values'] for frame in video_frames]
                video = [torch.stack(_temp_frames, dim=0)]
            else:
                # Fallback for empty frames
                print(f"WARNING: video {video_file} is empty.")
                video = []
        
        # Process STSG data if available
        stsg_data = None
        if 'situations' in sources[0]:
            # Here you would transform the STAR format STSG data to the format required by MotionEpic
            # This will depend on the exact format expected by the model
            stsg_data = self._process_stsg(sources[0]['situations'])
        
        # Process conversation data
        sources = self._preprocess_multimodal([e["conversations"] for e in sources])
        data_dict = self._preprocess_texts(sources, has_other_modality=True)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                           labels=data_dict["labels"][0])

        if 'input_video' in self.list_data_dict[i] and len(video) > 0:
            data_dict['video'] = video
            
        if stsg_data is not None:
            data_dict['stsg'] = stsg_data
            
        return data_dict
    
    def _read_video_with_timestamps(self, video_path, start, end, num_frames=8):
        """Read video using PyAV with specific time range"""
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            # Calculate frame indices based on timestamps
            frame_rate = video_stream.average_rate
            start_frame = int(start * frame_rate)
            end_frame = int(end * frame_rate)
            
            # Make sure we have enough frames
            if end_frame - start_frame < num_frames:
                end_frame = start_frame + num_frames
                
            # Sample evenly spaced frames
            indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
            
            frames = []
            container.seek(0)
            for i, frame in enumerate(container.decode(video=0)):
                if i > end_frame:
                    break
                if i >= start_frame and i in indices:
                    img = frame.to_image()
                    frames.append(img)
                    
            # Padding if necessary
            while len(frames) < num_frames:
                frames.append(Image.new('RGB', frames[0].size if frames else (224, 224)))
                
            return frames[:num_frames]  # Ensure we have exactly num_frames
            
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return []
    
    def _read_video(self, video_path, max_frames=8):
        """Read video without specific timestamps"""
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            total_frames = video_stream.frames
            
            # Sample evenly spaced frames
            indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    img = frame.to_image()
                    frames.append(img)
                if len(frames) >= max_frames:
                    break
                    
            return frames
            
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return []
    
    def _process_stsg(self, situations):
        """Convert STAR situation graphs to MotionEpic STSG format"""
        # This is a placeholder - you'll need to adapt this based on the exact format MotionEpic expects
        # For example, MotionEpic might expect a list of nodes and edges per frame
        
        frames = []
        for frame_id, situation in sorted(situations.items()):
            # Extract objects, relations, etc.
            rel_pairs = situation.get('rel_pairs', [])
            rel_labels = situation.get('rel_labels', [])
            
            # Here's a simplified conversion - you'll need to adapt this
            frame_data = {
                "frame_id": frame_id,
                "objects": [],  # Fill with objects from the situation
                "triplets": []  # Fill with subject-predicate-object triplets
            }
            
            # Add any objects from rel_pairs
            unique_objects = set()
            for pair in rel_pairs:
                unique_objects.update(pair)
            
            for obj_id in unique_objects:
                frame_data["objects"].append({"id": obj_id})
            
            # Add relationships
            for (subj, obj), rel in zip(rel_pairs, rel_labels):
                frame_data["triplets"].append((subj, rel, obj))
            
            frames.append(frame_data)
        
        return frames
    
    def _preprocess_multimodal(self, sources):
        """Adapt STAR data to the format expected by MotionEpic"""
        # This should do similar processing as in dataset_utils.py's preprocess_multimodal function
        for source in sources:
            for sentence in source:
                if DEFAULT_VIDEO_TOKEN in sentence['value']:
                    # Ensure video token is properly formatted
                    if not sentence['value'].startswith(DEFAULT_VIDEO_TOKEN):
                        sentence['value'] = DEFAULT_VIDEO_TOKEN + "\n" + sentence['value'].replace(DEFAULT_VIDEO_TOKEN, "").strip()
                        
                if DEFAULT_SG_TOKEN in sentence['value']:
                    # Similar processing for scene graph tokens
                    pass
        
        return sources
    
    def _preprocess_texts(self, sources, has_other_modality=True):
        """Process text inputs similar to dataset_utils.py's preprocess function"""
        # Here we need to call similar functions to what's in dataset_utils.py
        # This is a placeholder - in practice you would reuse the code from dataset_utils.py
        
        # You might need to import these functions or reimplement them
        from motionepic.dataset.dataset_utils import preprocess
        
        return preprocess(sources, self.tokenizer, has_other_modality=has_other_modality)