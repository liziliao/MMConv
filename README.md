


# [MMConv](https://liziliao.github.io/papers/2021sigir_mmconv.pdf): An Environment for Multimodal Conversational Search across Multiple Domains
Authors: [Lizi Liao](https://scholar.google.com.sg/citations?user=W2b08EUAAAAJ&hl=en), [Long Le Hong](https://github.com/LongLeCE), [Zheng Zhang](https://scholar.google.com.sg/citations?hl=en&user=S2bil1cAAAAJ), [Minlie Huang](https://scholar.google.com.sg/citations?hl=en&user=P1jPSzMAAAAJ), and [Tat-Seng Chua](https://scholar.google.com.sg/citations?hl=en&user=Z9DWCBEAAAAJ)

## Introduction
This work introduces a Multimodal Multi-domain Conversational search (MMConv) environment. It provides a large-scale multi-turn conversational corpus with dialogues spanning across several domains and modalities. Along which, there are also paired real user settings, structured venue database, annotated image repository as well as crowd-sourced knowledge base. More importantly, each dialogue is fully annotated with a sequence of dialogue belief states and corresponding system dialogue acts which is scarce in existing multimodal conversation corpora. 

MMConv can be used to develop individual system modules for conversational search following task-oriented dialogue research. On the other
hand, with over 5k fully annotated dialogues, MMConv also enables researchers to carry on end-to-end conversational modelling experiments. Accordingly, we provide a set of bench-marking results using current SOTA methods for various tasks, which may facilitate a lot of exciting ongoing research in the area.

 

## Table of Contents
- [Installation](#installation) 
- [Usage](#usage) 
    - [Preprocessing](#preprocessing)
    - [Benchmarks](#Benchmarks)
- [Citation](#citation)
- [License](#license)
 

## Installation

The package general requirements are

- Python >= 3.6
- Pytorch >= 1.2 (installation instructions [here](https://pytorch.org/))
- Transformers >= 2.5.1 (installation instructions [here](https://huggingface.co/transformers/))
 
The package can be installed by running the following command.  

```pip install -r requirements.txt```


## Usage
This section explains steps to preprocess MMConv dataset and training the model. 

### Preprocessing: 
It includes performing delexicaliztion, and creating dataset for various tasks.
```
run the blocks in generate_inputs.ipynb inside each method's folder
```
For example, for the multitask benchmark, each dialogue turn will be represented as a sequence, which contains previous user/system turns, belief, action, and delexicalized response

```
<|context|> <|user|> hi, i am looking for a place for dinner. could you help me find one? <|endofcontext|> <|belief|> menus dinner inform <|endofbelief|> <|action|> open span more information request <|endofaction|> <|response|> <|system|> sure, can you tell me more about the place you're looking for? <|endofresponse|>
```


### Benchmarks: 
We adopt several SOTA methods on the MMConv dataset. The original papers are listed as below:
-  Dialogue State Tracking    [DS-DST](https://arxiv.org/abs/1910.03544).
-  Conversation recommendation    [UMGR](https://www.aclweb.org/anthology/2020.coling-main.463/).
-  Response Generation    [MARCO](https://www.aclweb.org/anthology/2020.acl-main.638.pdf).
-  Joint Model across Multiple Tasks    [SimpleTOD](https://proceedings.neurips.cc/paper/2020/hash/e946209592563be0f01c844ab2170f0c-Abstract.html).

We organize the codes into separate folders for them. More details for training and evaluation can be found in corresponding folders.

## Citation
```
@inproceedings{{lizi2021mmconv,
  title={MMConv: An Environment for Multimodal Conversational Search across Multiple Domains},
  author={Liao, Lizi and Le Hong, Long and Zhang, Zheng and Huang, Minlie and Chua, Tat-Seng},
  booktitle={The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```


## License
The code is released under the MIT License - see [LICENSE](LICENSE) for details
