from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.generic_utils import read


class MMDialDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, i):
        context = MMDialDataset.extract(self.data[i], '<|context|>', keep_tokens=True)
        labels = self.data[i][len(context):]
        # belief = MMDialDataset.extract(labels, '<|belief|>')
        # action = MMDialDataset.extract(labels, '<|action|>')
        # response = MMDialDataset.extract(labels, '<|response|>')
        ret = self.tokenizer(self.data[i], truncation=True, return_tensors='pt')
        context_tokenized = self.tokenizer(context, truncation=True, return_tensors='pt')
        ret['context_input_ids'] = context_tokenized['input_ids']
        ret['context_attention_mask'] = context_tokenized['attention_mask']
        labels_tokenized = self.tokenizer(labels, truncation=True, return_tensors='pt')
        ret['labels'] = labels_tokenized['input_ids']
        ret['labels_len'] = ret['labels'].shape[-1]
        ret['id'] = i
        return ret

    @classmethod
    def get_token_text(cls, token):
        return token.replace('<', '').replace('>', '').replace('|', '').strip()

    @classmethod
    def extract(cls, text, begin_token, end_token=None, keep_tokens=False):
        end_token = end_token or f'<|endof{MMDialDataset.get_token_text(begin_token)}|>'
        begin_idx = text.find(begin_token)
        end_idx = text.find(end_token)
        if begin_idx == -1:
            return ''
        elif end_idx == -1:
            return text[begin_idx + len(begin_token):].strip() if not keep_tokens else text[begin_idx:]
        return text[begin_idx + len(begin_token): end_idx].strip() if not keep_tokens else text[begin_idx: end_idx + len(end_token)]

    @classmethod
    def create_data(cls, paths, tokenizer_or_transformer_model, split=(1,), shuffle=True):
        assert sum(split) == 1
        data = []
        for path in paths:
            data.extend(read(path))
        if isinstance(tokenizer_or_transformer_model, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_transformer_model)
        else:
            tokenizer = tokenizer_or_transformer_model
        if shuffle:
            random.shuffle(data)
        splits = []
        begin_idx = 0
        for i, s in enumerate(split):
            if i == len(split) - 1:
                end_idx = len(data)
            else:
                end_idx = int(begin_idx + len(data) * s)
            splits.append(MMDialDataset(data[begin_idx: end_idx], tokenizer=tokenizer))
            begin_idx = end_idx
        return splits[0] if len(split) == 1 else splits


if __name__ == '__main__':
    dataset = MMDialDataset.create_data(['resources/gpt2/resources/test.inp'], 'gpt2')
    for sample in dataset:
        print(sample)
        input()
