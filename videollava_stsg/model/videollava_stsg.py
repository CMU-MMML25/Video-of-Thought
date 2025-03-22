# Copyright 2023 Haotian Liu, 2024 MotionEpic Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from videollava.model.llava_arch import LlavaMetaModel
from videollava.model.multimodal_encoder.builder import build_image_tower, build_video_tower
# Add this import to support STSG
from motionepic.model.multimodal_encoder.builder import build_sg_encoder
from videollava.model.multimodal_projector.builder import build_vision_projector

from videollava_stsg.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, \
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, \
    DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, VIDEO_TOKEN_INDEX, SG_TOKEN_INDEX


class VideoLlavaSTSGMetaModel(LlavaMetaModel):
    def __init__(self, config):
        super(VideoLlavaSTSGMetaModel, self).__init__(config)

        # Additional initialization for STSG encoder
        if hasattr(config, "sg_encoder"):
            self.sg_encoder = build_sg_encoder(config)

    ### -------------------------- Video-LLAVA components --------------------------
    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower
    ### -----------------------------------------------------------------------------

    def get_sg_encoder(self):
        sg_encoder = getattr(self, 'sg_encoder', None)
        if type(sg_encoder) is list:
            sg_encoder = sg_encoder[0]
        return sg_encoder
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        # Call parent method to initialize image and video towers
        super().initialize_vision_modules(model_args, fsdp)
        
        # Additional initialization for STSG encoder
        if hasattr(model_args, "sg_encoder"):
            model_args.stsg_out_dim = self.config.hidden_size
            multimodal_tower = self.get_multimodal_tower()
            if multimodal_tower is not None:
                model_args.stsg_in_dim = multimodal_tower.hidden_size
            else:
                video_tower = self.get_video_tower()
                if video_tower is not None:
                    model_args.stsg_in_dim = video_tower.hidden_size
                else:
                    image_tower = self.get_image_tower()
                    if image_tower is not None:
                        model_args.stsg_in_dim = image_tower.hidden_size
            
            if self.get_sg_encoder() is None:
                sg_encoder = build_sg_encoder(model_args)
                if fsdp is not None and len(fsdp) > 0:
                    self.sg_encoder = [sg_encoder]
                else:
                    self.sg_encoder = sg_encoder
    
    # This is a helper property to maintain compatibility with MotionEpic
    def get_multimodal_tower(self):
        video_tower = self.get_video_tower()
        if video_tower is not None:
            return video_tower
        return self.get_image_tower()


class VideoLlavaSTSGMetaForCausalLM:
    def get_model(self):
        pass
    
    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()
    
    def get_sg_encoder(self):
        return self.get_model().get_sg_encoder()
    
    def encode_images(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_videos(self, videos):
        # Handle different video formats
        if videos.ndim == 5:  # [batch, channel, time, height, width]
            b, _, t, _, _ = videos.shape
            video_features = self.get_model().get_video_tower()(videos)
            video_features = self.get_model().mm_projector(video_features)
            return video_features
        elif videos.ndim == 4:  # [channel, time, height, width]
            videos = videos.unsqueeze(0)  # Add batch dimension
            video_features = self.get_model().get_video_tower()(videos)
            video_features = self.get_model().mm_projector(video_features)
            return video_features
    
    def encode_sg(self, sgs):
        sg_features = self.get_model().get_sg_encoder()(sgs)
        return sg_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images=None, videos=None, sgs=None
    ):
        # Get towers
        image_tower = self.get_model().get_image_tower()
        video_tower = self.get_model().get_video_tower()
        sg_encoder = self.get_model().get_sg_encoder()
        
        # Check if we need to process multimodal inputs
        no_images = images is None or (isinstance(images, list) and len(images) == 0)
        no_videos = videos is None or (isinstance(videos, list) and len(videos) == 0)
        no_sgs = sgs is None or (isinstance(sgs, list) and len(sgs) == 0)
        
        if ((image_tower is None and video_tower is None) or (no_images and no_videos)) and (sg_encoder is None or no_sgs):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        if input_ids.shape[1] == 1:
            if past_key_values is not None and not (no_images and no_videos and no_sgs):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # Process image/video features - adapted from LlavaMetaForCausalLM
        image_features = []
        if not no_images and image_tower is not None:
            if isinstance(images, list):
                image_features = [self.encode_images(img.unsqueeze(0))[0] for img in images]
            else:
                image_features = [self.encode_images(images)[0]]
        
        video_features = []
        if not no_videos and video_tower is not None:
            if isinstance(videos, list):
                for video in videos:
                    # Handle single frame or multi-frame
                    if video.ndim == 4:  # [C, T, H, W]
                        video = video.unsqueeze(0)  # [1, C, T, H, W]
                    
                    # Process each video
                    vid_features = self.encode_videos(video)
                    # If temporal dimension exists, separate them
                    if vid_features.ndim > 2:
                        for t in range(vid_features.shape[0]):
                            video_features.append(vid_features[t])
                    else:
                        video_features.append(vid_features)
            else:
                vid_features = self.encode_videos(videos)
                if vid_features.ndim > 2:
                    for t in range(vid_features.shape[0]):
                        video_features.append(vid_features[t])
                else:
                    video_features.append(vid_features)
        
        # Process SG features
        sg_features = []
        if not no_sgs and sg_encoder is not None:
            if isinstance(sgs, list):
                sg_features = [self.encode_sg(sg.unsqueeze(0))[0] for sg in sgs]
            else:
                sg_features = [self.encode_sg(sgs)[0]]
        
        # Use defaults for missing values
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # Process each example in the batch - following MotionEpic's approach
        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_video_idx = 0
        cur_sg_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_videos = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()
            num_sgs = (cur_input_ids == SG_TOKEN_INDEX).sum()
            
            if num_images == 0 and num_videos == 0 and num_sgs == 0:
                # No multimodal tokens, just process text
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue
            
            # Find positions of all special tokens
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            video_token_indices = torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist()
            sg_token_indices = torch.where(cur_input_ids == SG_TOKEN_INDEX)[0].tolist()
            
            # Combine all special token indices and sort them
            special_token_indices = image_token_indices + video_token_indices + sg_token_indices
            special_token_indices.sort()
            special_token_indices = [-1] + special_token_indices + [cur_input_ids.shape[0]]
            
            # Split text by special tokens
            cur_input_ids_segments = []
            cur_labels_segments = []
            cur_labels = labels[batch_idx]
            
            for i in range(len(special_token_indices) - 1):
                start_idx = special_token_indices[i] + 1
                end_idx = special_token_indices[i + 1]
                cur_input_ids_segments.append(cur_input_ids[start_idx:end_idx])
                cur_labels_segments.append(cur_labels[start_idx:end_idx])
            
            # Get embeddings for text segments
            split_sizes = [x.shape[0] for x in cur_labels_segments]
            if len(torch.cat(cur_input_ids_segments)) > 0:  # Make sure there's text to embed
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_segments))
                cur_input_embeds_segments = torch.split(cur_input_embeds, split_sizes, dim=0)
            else:
                cur_input_embeds_segments = [torch.tensor([], device=cur_labels.device).reshape(0, self.config.hidden_size)]
            
            # Combine text embeddings with image/video/sg embeddings
            cur_new_input_embeds = []
            cur_new_labels = []
            
            for i in range(len(special_token_indices) - 1):
                # Add text segment
                if i < len(cur_input_embeds_segments):
                    cur_new_input_embeds.append(cur_input_embeds_segments[i])
                    cur_new_labels.append(cur_labels_segments[i])
                
                # Add special token embedding if not at the end
                if i < len(special_token_indices) - 1:
                    token_idx = special_token_indices[i + 1]
                    if token_idx in image_token_indices:
                        # Add image embedding
                        if cur_image_idx < len(image_features):
                            cur_mm_features = image_features[cur_image_idx]
                            cur_image_idx += 1
                            cur_new_input_embeds.append(cur_mm_features)
                            cur_new_labels.append(
                                torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, 
                                          device=cur_labels.device, dtype=cur_labels.dtype)
                            )
                    elif token_idx in video_token_indices:
                        # Add video embedding
                        if cur_video_idx < len(video_features):
                            cur_mm_features = video_features[cur_video_idx]
                            cur_video_idx += 1
                            cur_new_input_embeds.append(cur_mm_features)
                            cur_new_labels.append(
                                torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, 
                                          device=cur_labels.device, dtype=cur_labels.dtype)
                            )
                    elif token_idx in sg_token_indices:
                        # Add scene graph embedding
                        if cur_sg_idx < len(sg_features):
                            cur_sg_feature = sg_features[cur_sg_idx]
                            cur_sg_idx += 1
                            cur_new_input_embeds.append(cur_sg_feature)
                            cur_new_labels.append(
                                torch.full((cur_sg_feature.shape[0],), IGNORE_INDEX, 
                                          device=cur_labels.device, dtype=cur_labels.dtype)
                            )
            
            # Concatenate all embeddings for this example
            if len(cur_new_input_embeds) > 0:
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)
                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
            else:
                # Fallback if no embeddings were added
                dummy_embed = self.get_model().embed_tokens(cur_input_ids[:1])
                new_input_embeds.append(dummy_embed)
                new_labels.append(labels[batch_idx][:1])
        
        # Truncate sequences to max length as embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        
        # Pad sequences to the same length
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(
                    torch.cat((
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        cur_new_embed
                    ), dim=0)
                )
                if cur_len > 0:
                    new_labels_padded[(i), -cur_len:] = cur_new_labels
                    attention_mask[(i), -cur_len:] = True
                    position_ids[(i), -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(
                    torch.cat((
                        cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                    ), dim=0)
                )
                if cur_len > 0:
                    new_labels_padded[(i), :cur_len] = cur_new_labels
                    attention_mask[(i), :cur_len] = True
                    position_ids[(i), :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        if _position_ids is None:
            position_ids = None
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # Initialize standard VideoLLaVA tokens
        super().initialize_vision_tokenizer(model_args, tokenizer)
        
        # Add STSG token - from MotionEpic
        signal_token_list = []
        signal_token_list.extend(["<scene_graph>"])
        
        num_new_tokens = tokenizer.add_tokens(signal_token_list, special_tokens=True)
        print(f"Adding {num_new_tokens} new tokens to the tokenizer.")
        self.resize_token_embeddings(len(tokenizer))
        
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data
            
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
            
            if model_args.tune_mm_input_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

