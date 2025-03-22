# Copyright (c) 2024 torchtorch Authors. All Rights Reserved.
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

from dataclasses import dataclass
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, GenerationConfig

from transformers.modeling_outputs import ModelOutput
from transformers.generation.utils import GenerateOutput

# from ..motionepic_arch import VideoLLAVASTSGMetaForCausalLM, VideoLLAVASTSGMetaModel
from videollava_stsg.model.videollava_stsg import VideoLlavaSTSGMetaModel, VideoLlavaSTSGMetaForCausalLM

from transformers import StoppingCriteria, StoppingCriteriaList

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
   

__all__ = [
    "VideoLLAVASTSGLlamaModel",
    "VideoLLAVASTSGLlamaForCausalLM",
]


class VideoLlavaSTSGConfig(LlamaConfig):
    model_type = "video_llava_stsg"


class VideoLlavaSTSGLlamaModel(VideoLlavaSTSGMetaModel, LlamaModel):
    config_class = VideoLlavaSTSGConfig

    def __init__(self, config: LlamaConfig):
        super(VideoLlavaSTSGLlamaModel, self).__init__(config)


class VideoLlavaSTSGLlamaForCausalLM(LlamaForCausalLM, VideoLlavaSTSGMetaForCausalLM):
    config_class = VideoLlavaSTSGConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VideoLlavaSTSGLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        videos: Optional[torch.FloatTensor] = None,
        sgs: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            # VideoLLaVA compatibility: if images is provided but videos is not,
            # convert images to videos format
            if images is not None and videos is None:
                videos = images
                
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                videos,
                sgs
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Capture multimodal inputs from kwargs
        images = kwargs.pop("images", None)
        videos = kwargs.pop("videos", None)
        sgs = kwargs.pop("sgs", None)
        
        # VideoLLaVA compatibility: if images is provided but videos is not,
        # use images as videos
        if images is not None and videos is None:
            videos = images
        
        # Get base inputs from parent
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        
        # Add multimodal inputs
        if images is not None:
            inputs['images'] = images
        if videos is not None:
            inputs['videos'] = videos
        if sgs is not None:
            inputs['sgs'] = sgs
        
        # In MotionEpic, the _get_generation method processes the embeddings
        # We'll do a similar processing here
        if inputs_embeds is None and past_key_values is None and (images is not None or videos is not None or sgs is not None):
            # Process the inputs through prepare_inputs_labels_for_multimodal to get embeddings
            (_, new_position_ids, new_attention_mask, new_past_key_values, new_inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                input_ids, 
                inputs.get('position_ids', None),
                inputs.get('attention_mask', None),
                None, 
                None,
                images,
                videos,
                sgs
            )
            
            # Update the inputs with processed values
            if new_inputs_embeds is not None:
                inputs['inputs_embeds'] = new_inputs_embeds
                inputs['input_ids'] = None  # Clear input_ids as we're using inputs_embeds
            
            if new_attention_mask is not None:
                inputs['attention_mask'] = new_attention_mask
                
            if new_position_ids is not None:
                inputs['position_ids'] = new_position_ids
                
        return inputs
        
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        videos: Optional[torch.FloatTensor] = None,
        sgs: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Overrides the generate method to handle multimodal inputs.
        """
        # VideoLLaVA compatibility: if images is provided but videos is not,
        # use images as videos
        if images is not None and videos is None:
            videos = images
            
        # Prepare inputs
        if inputs_embeds is None and input_ids is not None and \
           (images is not None or videos is not None or sgs is not None):
            # Get generation inputs, which sets up input_embeds if needed
            inputs = self.prepare_inputs_for_generation(
                input_ids, 
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                images=images,
                videos=videos,
                sgs=sgs,
                use_cache=kwargs.get("use_cache", True)
            )
            
            # Extract the inputs we need for generation
            input_ids = inputs.get("input_ids", None)
            inputs_embeds = inputs.get("inputs_embeds", None)
            attention_mask = inputs.get("attention_mask", None)
            position_ids = inputs.get("position_ids", None)
            past_key_values = inputs.get("past_key_values", None)
        
        # Ensure we don't pass None values to super().generate
        generate_kwargs = {
            "use_cache": kwargs.pop("use_cache", True),
            "return_dict_in_generate": kwargs.pop("return_dict_in_generate", True),
            **kwargs
        }
        
        if input_ids is not None:
            generate_kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            generate_kwargs["inputs_embeds"] = inputs_embeds
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            generate_kwargs["position_ids"] = position_ids
        if past_key_values is not None:
            generate_kwargs["past_key_values"] = past_key_values
        
        # Call LlamaForCausalLM's generate method with properly prepared inputs
        return super().generate(**generate_kwargs)

# Register configs and models
AutoConfig.register("video_llava_stsg", VideoLlavaSTSGConfig)
AutoModelForCausalLM.register(VideoLlavaSTSGConfig, VideoLlavaSTSGLlamaForCausalLM)
