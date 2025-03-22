import torch
from videollava_stsg.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava_stsg.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# Disable torch initialization
disable_torch_init()

# Configuration
video_path = '../STAR/data/Charades_v1_480/0A8CF.mp4'
inp = 'Why is this video funny?'
model_path = 'LanguageBind/Video-LLaVA-7B'
cache_dir = 'cache_dir'
device = 'cuda'

# Load the model
load_4bit, load_8bit = True, False
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, _ = load_pretrained_model(
    model_path, 
    None, 
    model_name, 
    load_8bit, 
    load_4bit, 
    device=device, 
    cache_dir=cache_dir,
    use_stsg=True  # Enable STSG support
)

# Use video processor
video_processor = processor['video']
conv_mode = "llava_v1"
conv = conv_templates[conv_mode].copy()
roles = conv.roles

# Process video
video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
if isinstance(video_tensor, list):
    tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
else:
    tensor = video_tensor.to(model.device, dtype=torch.float16)

print(f"{roles[1]}: {inp}")

# Get video token count from model
try:
    video_token_count = model.get_video_tower().config.num_frames
except:
    # Fallback if the above fails
    video_token_count = 8  # Default fallback

# Prepare input with video tokens (exactly like VideoLLaVA)
inp_with_tokens = ' '.join([DEFAULT_IMAGE_TOKEN] * video_token_count) + '\n' + inp
conv.append_message(conv.roles[0], inp_with_tokens)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# Tokenize input
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

# Set up stopping criteria
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

# Generate response
with torch.inference_mode():
    generation_output = model.generate(
        input_ids,
        images=tensor,  # For VideoLLaVA compatibility, pass as images
        do_sample=True,
        temperature=0.1,
        max_new_tokens=1024,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # Check what type of output we got
    if hasattr(generation_output, "sequences"):
        # It's a GenerationOutput object
        output_ids = generation_output.sequences
    elif isinstance(generation_output, torch.Tensor):
        # It's directly a tensor of output IDs
        output_ids = generation_output
    else:
        # Try to convert it to a tensor
        try:
            output_ids = generation_output[0]
        except:
            print(f"Unexpected output type: {type(generation_output)}")
            print(f"Output content: {generation_output}")
            raise

# Decode and print the response (adjust indexing based on output type)
outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
print(outputs)