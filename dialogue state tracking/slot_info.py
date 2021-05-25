from utils.generic_utils import read
from dataloader import extract
from collections import defaultdict
from utils.json_utils import save

slot_names = {'wheelchair accessible', 'reservations', 'restroom', 'smoking', 'credit cards', 'outdoor seating', 'parking', 'music', 'wi-fi', 'dining options', 'drinks', 'venuescore', 'menus', 'price', 'venueneigh', 'venuename', 'telephone', 'venueaddress', 'img_gt', 'open span'}

slot_values = defaultdict(set)

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
