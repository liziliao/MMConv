# Joint Model across Multiple Tasks
This is code of model for Joint Model across Multiple Tasks  [SimpleTOD](https://proceedings.neurips.cc/paper/2020/hash/e946209592563be0f01c844ab2170f0c-Abstract.html).
It corresponds to model in article *MMConv: an Environment for Multimodal Conversational Search across Multiple Domains*, section 5.4. To reproduce results shown in paper, follow the steps below:

### input generation
```run the blocks in generate_inputs.ipynb
```
The preprocessed inputs will be stored in folder /resources .

### model training
```sh train_mmdial.sh $CUDA_VISIBLE_DEVICES $nproc_per_node $MODEL $MODEL_NAME $BATCH $save_total_limit
```

You need to assign parameters above in the very sequence. Particularly, make sure $nproc_per_node, the number of GPUs used when training parallelly is not larger than number of $CUDA_VISIBLE_DEVICES. $MODEL is the name of model group used as backbone, and $MODEL_NAME is the name of specific model or the path to that model predownloaded. $save_total_limit is the number of maximum checkpoints saved in sliding updating manner.

One runnable example is like this:
```sh train_mmdial.sh 0,1,2,3 4 gpt2 ./example_model 2 5
```

### model evaluation
```python generate_simpletod.py $MODEL $BATCH $checkpoint
```
Here $checkpoint is one of model files saved in ./checkpoints. Note the numbers within name of checkpoint files are global steps trained till the checkpoint in **main process** only.

