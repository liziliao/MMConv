# Joint Model across Multiple Tasks
This is code of model for Dialogue State Tracking task  [DS-DST](https://arxiv.org/abs/1910.03544).
It corresponds to model in article *MMConv: an Environment for Multimodal Conversational Search across Multiple Domains*, section 5.1. To reproduce results shown in paper, follow the steps below:

### input generation
```run the blocks in generate_inputs.ipynb```

The preprocessed inputs will be stored in folder /resources.

Note: For dst, after generating {train,val,test}.dst, run
```python slot_info.py```
to obtain slot_values.json in the folder /resources.

### model training
Example command: 
```python train_rg.py {BATCH_SIZE}```

Models are saved in checkpoint/, predictions saved in prediction

Or, you may simply run the train_dst.ipynb file.

### Training EfficientNet to get image predictions
Put the images in data/images
Run split_img.py to generate data splits, model_en.py to train efficientnet, note that the EMA weights are also saved

