from utils.generic_utils import read
#from dataloader import extract
from collections import defaultdict
from utils.json_utils import save
import re

slot_names = {'wheelchair accessible', 'reservations', 'restroom', 'smoking', 'credit cards', 'outdoor seating', 'parking', 'music', 'wi-fi', 'dining options', 'drinks', 'venuescore', 'menus', 'price', 'venueneigh', 'venuename', 'telephone', 'venueaddress', 'img_gt', 'open span'}

slot_values = defaultdict(set)

token_matcher = re.compile(r'<\|[a-zA-Z]+\|>')

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

def has_token(text):
    return next_token(text) is not None

def next_token(text):
    result = token_matcher.search(text)
    return result if result is None else result[0]


def get_token_text(token):
    return token.replace('<', '').replace('>', '').replace('|', '').replace('[', '').replace(']', '')

input_format = 'dst'
for split in ['train', 'test', 'val']:
    data = read(f'resources/{split}.{input_format}')
    for sample in data:
        belief = extract(sample, '<|belief|>')[0]
        if belief:
            for slot in belief.split('; '):
                slot_split = slot.split()
                action = slot_split[-1]
                slot_name = slot_split[0] if slot_split[0] in slot_names else ' '.join(slot_split[:2])
                assert slot_name in slot_names
                value = ' '.join(slot_split[len(slot_name.split()): -1])
                if not value:
                    value = None
                if value is not None:
                    slot_values[slot_name].add(value)

slot_values_sorted = {}
for k in sorted(slot_values.keys()):
    slot_values_sorted[k] = sorted(slot_values[k])
save(slot_values_sorted, 'resources/slot_values.json')
