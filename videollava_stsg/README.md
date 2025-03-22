# VideoLLAVA-STSG

## Installation

Install dependencies:

```
pip install -r requirements_test.txt
```


> Note: `MotionEpic` requires `transformers>=4.37.2`, while `VideoLLAVA` requires `transformers<=4.33.0`.  
> After testing, `transformers==4.37.2` is the best option. However, you need to manually add the definition for `_expand_mask()` in `transformers/models/clip/modeling_clip.py`.  
> Please follow this post: https://github.com/salesforce/LAVIS/issues/571

## Inference

To run inference on VideoLLAVA:

```
CUDA_VISIBLE_DEVICES=0 python inference_videollava.py
```

To run inference on VideoLLAVA-STSG:
```
CUDA_VISIBLE_DEVICES=0 python inference_videollava_stsg.py
```