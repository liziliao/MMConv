import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.generic_utils import read, match
from utils.json_utils import load
import os
import random
import re
from torchvision.transforms import transforms
from PIL import Image
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from timm.data.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval
from collections import defaultdict

# **use your own image class prediction results here**
img_classes = load('image_classes.json')

img2class = {v: k for k, vs in img_classes.items() for v in vs}
class2id = {k: i for i, k in enumerate(img_classes.keys())}

# MMDial
slot_values = load('resources/slot_values.json')
acts = ["greet", "inform", "request", "recommend", "negate", "confirm", "bye", "others", "doncare"]
act_order = {
    "negate": 0,
    "doncare": 1,
    "confirm": 2,
    "recommend": 3,
    "inform": 4,
    "request": 5,
    "others": 6,
    "greet": 7,
    "bye": 8
}

# # WOZ2.0
# slot_values = load('resources/slot_values_woz20.json')
# acts = ["inform", "request"]

# # DSTC2
# slot_values = load('resources/slot_values_dstc2.json')
# acts = ['confirm', 'deny', 'inform', 'request']

act_ids = {a: i for i, a in enumerate(acts)}

for slot, values in slot_values.items():
    if len(values) <= 2: # yes/no
        for value in ['yes', 'no']:
            if value not in values:
                slot_values[slot].append(value)
        slot_values[slot] = sorted(slot_values[slot])
slot_values_keys = sorted(set(slot_values.keys()).difference(['img_gt']))
slot_idxes = {x: i for i, x in enumerate(slot_values_keys)}

slot_tokens = defaultdict(list)

open_slots = {'open span', 'venuename', 'venueneigh', 'venueaddress', 'telephone', 'name', 'phone', 'postcode'}

token_matcher = re.compile(r'<\|[a-zA-Z]+\|>')

phone_matcher = re.compile('\d{5} \d{6}')
postcode_matcher = re.compile('[A-Z][.][A-Z] \d, \d [A-Z][.][A-Z]')
telephone_matcher = re.compile('(65|65 |[+]65|[+]65 )?\d{4} ?\d{4}')
matchable_slots = {
    'telephone': telephone_matcher,
    'phone': phone_matcher,
    'postcode': postcode_matcher
}

slot_correction = {
    'venuenaddress': 'venueaddress',
    'venuenname': 'venuename',
    'venueneight': 'venueneigh',
    'menu': 'menus',
    'drink': 'drinks',
    'musics': 'music',
    'reservation': 'reservations',
    'credit card': 'credit cards',
    'outdoor seatings': 'outdoor seating',
    'dining option': 'dining options',
    'wifi': 'wi-fi'
}

for os_slot in ['openspan', 'open psan', 'opne span', 'open open', 'opan span', 'open sapn', 'open span:', 'oprn span', 'openn span', 'open spicy', 'opens span', 'open spam', 'oepn span']:
    slot_correction[os_slot] = 'open span'


def url2img(url_or_img):
    begin_idx = url_or_img.rfind('/')
    if begin_idx != -1:
        return url_or_img[begin_idx + 1:]
    return url_or_img


def has_token(text):
    return next_token(text) is not None


def next_token(text):
    result = token_matcher.search(text)
    return result if result is None else result[0]


def get_token_text(token):
    return token.replace('<', '').replace('>', '').replace('|', '').replace('[', '').replace(']', '')


def extract(text, begin_token, end_token=None, no_token_in_between=True):
    end_token = end_token or f'<|endof{get_token_text(begin_token)}|>'
    begin_idx = text.find(begin_token)
    if begin_idx == -1:
        return '', None
    begin_with_len = begin_idx + len(begin_token)
    end_idx = text[begin_with_len:].find(end_token)
    if end_idx == -1:
        return '', None
    end_idx += begin_with_len
    next_token_ = next_token(text[begin_with_len:])
    if not no_token_in_between or next_token_ == end_token:
        return text[begin_with_len: end_idx].strip(), begin_idx
    recurse_result = extract(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between)
    return recurse_result[0], (recurse_result[1] + begin_with_len) if recurse_result[1] is not None else None


def remove(text, begin_token, end_token=None, no_token_in_between=True, remove_begin_token=True, remove_end_token=True):
    end_token = end_token or f'<|endof{get_token_text(begin_token)}|>'
    begin_idx = text.find(begin_token)
    if begin_idx == -1:
        return text
    begin_with_len = begin_idx + len(begin_token)
    end_idx = text[begin_with_len:].find(end_token)
    if end_idx == -1:
        return text
    end_idx += begin_with_len
    next_token_ = next_token(text[begin_with_len:])
    if not no_token_in_between or next_token_ == end_token:
        end_with_len = end_idx + len(end_token)
        return text[:(begin_idx if remove_begin_token else begin_with_len)].strip() + ' ' + text[(end_with_len if remove_end_token else end_idx):].strip()
    return text[:begin_with_len] + remove(text[begin_with_len:], begin_token, end_token=end_token, no_token_in_between=no_token_in_between, remove_begin_token=remove_begin_token, remove_end_token=remove_end_token)


def read_slot(slot):
    slot_split = slot.split()
    act = slot_split[-1]
    name = slot_split[0] if slot_split[0] in slot_values else ' '.join(slot_split[:2])
    value = ' '.join(slot_split[len(name.split()): -1])
    if not value:
        value = None
    return name, value, act


class BaseDataset(Dataset):
    def __init__(self, data, training, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.training = training

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, i):
        raise NotImplementedError()

    @classmethod
    def read_data(cls, paths):
        raise NotImplementedError()

    @classmethod
    def create_data(cls, paths, split=(1,), training=(True,), shuffle=True, **kwargs):
        assert sum(split) == 1
        if isinstance(training, bool):
            training = [training for _ in range(len(split))]
        elif len(training) == 1 and len(split) > 1:
            training = [training[0] for _ in range(len(split))]
        assert len(split) == len(training)
        if isinstance(paths, str):
            paths = [paths]
        print('Preparing data...')
        data = cls.read_data(paths)
        if shuffle:
            random.shuffle(data)
        splits = []
        begin_idx = 0
        for i, s in enumerate(split):
            if i == len(split) - 1:
                end_idx = len(data)
            else:
                end_idx = int(begin_idx + len(data) * s)
            splits.append(cls(data[begin_idx: end_idx], training=training[i], **kwargs))
            begin_idx = end_idx
        return splits[0] if len(split) == 1 else splits


class TextDataset(BaseDataset):
    def __init__(
        self,
        data,
        training,
        tokenizer_or_transformer_model,
        max_len=1024,
        end_token=' <|endofresponse|>',
        remove_tokens={
            '<|imagesource|>': {'<|system|>', '<|user|>', '<|endofcontext|>', '<|endofresponse|>'}
        },
        split_token='<|endofcontext|>',
        task='rg',
        data_var_token='<|belief|>',
        split_data_var='; ',
        reduce_slots_without_values=True,
        **kwargs
    ):
        super().__init__(data, training, **kwargs)
        if isinstance(tokenizer_or_transformer_model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_transformer_model)
        else:
            self.tokenizer = tokenizer_or_transformer_model
        self.end_tokens = self.tokenizer(end_token, truncation=True, max_length=1024, return_tensors='pt')['input_ids']
        self.remove_tokens = remove_tokens
        self.split_token = split_token
        self.max_len = max_len
        self.data_var_token = data_var_token
        self.split_data_var = split_data_var
        self.reduce_slots_without_values = reduce_slots_without_values
        self.task = task

    def update_task(self):
        if self.task == 'dst':
            self.len_map = {}
            self.item_idx_map = {}
            self.sample_slot_map = {}
            total = 0
            for i, sample in enumerate(self.data):
                last_total = total
                # slots = extract(sample, self.data_var_token)[0]
                # slot_names = set()
                # if slots:
                #     slots = slots.split(self.split_data_var)
                #     for slot in slots:
                #         name, value, act = read_slot(slot)
                #         if name != 'img_gt':
                #             slot_names.add(name)
                # total += len(slot_names)

                slots = extract(sample, self.data_var_token)[0]
                if slots:
                    slots = [s for s in slots.split(self.split_data_var) if not s.startswith('img_gt')]
                else:
                    slots = []
                slot_names_with_values = set()
                slot_names_without_values = set()
                for slot in slots:
                    name, value, act = read_slot(slot)
                    if value is not None and act is not None:
                        slot_names_with_values.add(name)
                slot_names_without_values = set(slot_values_keys).difference(slot_names_with_values)
                num_slots_with_values = len(slot_names_with_values)
                if self.reduce_slots_without_values:
                    # Reducing num of slots without values
                    num_slots_without_values = min(max(len(slot_names_with_values), 2), len(slot_names_without_values))
                else:
                    num_slots_without_values = len(slot_names_without_values)
                total += num_slots_with_values + num_slots_without_values

                self.len_map[i] = total
                for j in range(last_total, total):
                    self.item_idx_map[j] = i
                temp_idx = 0
                for name in sorted(slot_names_with_values):
                    self.sample_slot_map[last_total + temp_idx] = slot_idxes[name]
                    temp_idx += 1
                for name in sorted(random.sample(slot_names_without_values, num_slots_without_values)):
                    self.sample_slot_map[last_total + temp_idx] = slot_idxes[name]
                    temp_idx += 1

            for slot in slot_values_keys:
                if slot not in slot_tokens and slot not in open_slots:
                    for candidate in slot_values[slot]:
                        slot_tokens[slot].append(self.tokenizer(candidate, return_tensors='pt')['input_ids'])

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'task':
            self.update_task()

    @classmethod
    def read_data(cls, paths):
        data = []
        for path in tqdm(paths):
            data.extend(read(path))
        return data

    def __len__(self):
        if self.task == 'dst':
            return self.len_map[len(self.data) - 1]
        return super().__len__()

    def __getitem__(self, i):
        if self.task == 'dst':
            raw_sample = self.data[self.item_idx_map[i]]
        else:
            raw_sample = self.data[i]
        for remove_token, end_tokens in self.remove_tokens.items():
            end_tokens = deepcopy(end_tokens)
            while end_tokens:
                for end_token in list(end_tokens):
                    img_src, _ = extract(raw_sample, remove_token, end_token=end_token)
                    if not img_src:
                        end_tokens.discard(end_token)
                    else:
                        raw_sample = remove(raw_sample, remove_token, end_token=end_token, remove_end_token=False)
        split_idx = raw_sample.rindex(self.split_token) + len(self.split_token)
        if self.task == 'rg':
            if self.training:
                sample = self.tokenizer(raw_sample, truncation=True, max_length=1024, return_tensors='pt')
            else:
                sample = self.tokenizer(raw_sample[:split_idx], truncation=True, max_length=1024, return_tensors='pt')
            sample['response'] = self.tokenizer(raw_sample[split_idx:], truncation=True, max_length=1024, return_tensors='pt')['input_ids']
            sample['response_begin'] = sample['input_ids'].shape[-1] - sample['response'].shape[-1]
            sample['response_end'] = sample['input_ids'].shape[-1]
            sample['end_tokens'] = self.end_tokens
            sample['max_len'] = self.max_len
        else: # dst
            slots = extract(raw_sample, self.data_var_token)[0]
            if slots:
                slots = [s for s in slots.split(self.split_data_var) if not s.startswith('img_gt')]
            else:
                slots = []
            idx = self.item_idx_map[i]
            slot_idx = self.sample_slot_map[i]

            slot_by_idx = slot_values_keys[slot_idx]

            has_value = False
            act2values = defaultdict(list)

            for slot in slots:
                name, value, act = read_slot(slot)
                if name == slot_by_idx:
                    has_value = has_value or value is not None
                    if act is not None and value is not None:
                        act2values[act].append(value)
            act2values = sorted(act2values.items(), key=lambda x: act_order[x[0]])

            name = slot_by_idx
            #print("name", name) 
            sample_str = raw_sample[:split_idx]
            tokenized_slot = ['[CLS]'] + self.tokenizer.tokenize(name) + ['[SEP]']
            tokenized_sample = self.tokenizer.tokenize(sample_str) + ['[SEP]']
            tokenized_sample = tokenized_slot + tokenized_sample[-(512 - len(tokenized_slot)):]
            #print("tokenized_sample", tokenized_sample)
            tokenized_sample_tensor = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(tokenized_sample))
            sample = {
                'input_ids': tokenized_sample_tensor,
                'token_type_ids': torch.cat([torch.zeros((len(tokenized_slot),), dtype=int), torch.ones((tokenized_sample_tensor.shape[0] - len(tokenized_slot),), dtype=int)]),
                'attention_mask': torch.ones_like(tokenized_sample_tensor)
            }

            # Span
            spans = set()
            if name in open_slots:
                if name == 'open span':
                    if act2values:
                        for value_ in act2values[0][1]:
                            spans.add(value_)
                elif has_value:
                    if name in matchable_slots:
                        for m in matchable_slots[name].finditer(sample_str.lower()):
                            spans.add(m[0])
                    else:
                        if act2values:
                            for value_ in act2values[0][1]:
                                matches = match(sample_str.lower(), value_, thresh_abs=int(max(1, 0.35*len(value_))), thresh_r=0.55, text_len_delta=[int(max(-8, min(-1, -0.3*len(value_.split())))), int(min(8, max(1, 0.3*len(value_.split()))))], return_thresh=1, sorted=True)
                                if matches:
                                    spans.add(matches[0][0])
                                # else:
                                #     print(sample_str)
                                #     print(value)
                                #     raise Exception()

            span_label = torch.zeros_like(sample['input_ids'], dtype=int)
            tokenized_sample_str = ' '.join(tokenized_sample)
            # open_span_count = 0
            for span in spans:
                tokenized_span = self.tokenizer.tokenize(span)
                tokenized_span_str = ' '.join(tokenized_span)
                idxs = [m.start() for m in re.finditer(re.escape(tokenized_span_str), tokenized_sample_str)]
                for j in idxs:
                    # open_span_count += 1
                    begin_idx = len(tokenized_sample_str[:j].split())
                    span_label[begin_idx] = 2
                    span_label[begin_idx + 1: begin_idx + len(tokenized_span)] = 1
            sample['span'] = span_label
            #print("span_label", span_label)
            # Action
            sample['action'] = act_ids[act2values[0][0]] if len(act2values) else -1

            # Gate
            sample['gate'] = int(has_value or (sample['action'] != -1 and (len(spans) > 0 or len(act2values) > 0)))

            # Slot
            sample['slot'] = slot_idx
            sample['slot value'] = -1 if (not act2values or slot_by_idx in open_slots) else slot_values[slot_by_idx].index(act2values[0][1][-1])

            # sample['slots'] = slot_options_lower

        if self.task == 'dst':
            sample['id'] = [self.item_idx_map[i], slot_idx]
        else:
            sample['id'] = i
        sample['input_ids_len'] = sample['input_ids'].shape[0]
        return sample


class ImageDataset(BaseDataset):
    def __init__(self, data, training, image_size,
                 mean=[0.49810758, 0.43045932, 0.35550067],
                 std=[0.24467874, 0.24427865, 0.24221295],
                 auto_augment=None,
                 **kwargs):
        super().__init__(data, training, **kwargs)
        if mean is None or std is None:
            self.transform = transforms.Compose([
                # transforms.Resize(image_size, interpolation=Image.BICUBIC),
                transforms.ToTensor()
            ])
            mean, std = self.get_norm_params(data, image_size)
        if self.training:
            self.augmentor = transforms_imagenet_train(
                img_size=image_size,
                scale=[0.08, 1],
                ratio=[3/4, 4/3],
                hflip=0.5,
                vflip=0,
                color_jitter=0.4,
                auto_augment=auto_augment,
                interpolation='random',
                use_prefetcher=False,
                mean=mean,
                std=std,
                re_prob=0.2,
                re_mode='pixel',
                re_count=1,
                re_num_splits=0,
                separate=False,
            )
        else:
            self.augmentor = transforms_imagenet_eval(
                img_size=image_size,
                crop_pct=1.0,
                interpolation='bicubic',
                use_prefetcher=False,
                mean=mean,
                std=std
            )

    def get_mean_and_std(self, path, dim=(1, 2)):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return torch.stack((img.mean(dim), img.std(dim)))

    def get_norm_params(self, dataset, image_size, workers=8):
        p = ThreadPool(workers)
        print('Calculating mean and std...')
        result = torch.stack(p.map(self.get_mean_and_std, tqdm(dataset)))
        mean, std = torch.mean(result, dim=0).numpy()
        print('Mean: {}\nStd: {}'.format(mean, std))
        return mean, std

    @classmethod
    def read_data(cls, paths):
        data = []
        for path in tqdm(paths):
            data.extend(read(path))
        return data

    def __getitem__(self, i):
        path = self.data[i]
        img = Image.open(path).convert('RGB')
        img = self.augmentor(img)
        label = class2id[img2class[path]]
        return {
            'image': img,
            'label': label,
            'id': i
        }


if __name__ == '__main__':
    test = ResponseGenerationDataset.create_data('resources/train.dialogpt', 'microsoft/DialoGPT-small', split=(1,), training=False, shuffle=False)
    print(test[15359])
