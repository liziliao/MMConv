import sys
from utils.json_utils import load
from utils.generic_utils import read
from collections import defaultdict
from dataloader import slot_values, slot_values_keys, open_slots, TextDataset, acts
import numpy as np
from tqdm import tqdm

score = defaultdict(int)
correct = defaultdict(bool)
total = 0

split_token = '<|endofcontext|>' 

include_open_span = True

all_samples = TextDataset.create_data('resources/test.dst', tokenizer_or_transformer_model='bert-base-uncased', split=(1,), training=False, shuffle=False, task='dst', data_var_token='<|belief|>', split_data_var='; ', reduce_slots_without_values=False)

# print(slot_values)
last_idx = None

all_predictions = load(sys.argv[1])
with open('pred_beliefs.txt', 'w+') as pf, open('gt_beliefs.txt', 'w+') as gf:
    for i, (idx, predictions) in enumerate(tqdm(all_predictions.items())):
        idx, slot_idx = eval(idx)
        slot = slot_values_keys[slot_idx]
        if idx != last_idx:
            if last_idx is not None:
                # print(input_text)
                # print(f'Predicted bstate: {bstate_pred}')
                # print(f'Groundtruth bstate: {bstate_gt}')
                # input('\n\n')
                pf.write('; '.join(bstate_pred) + '\n')
                gf.write('; '.join(bstate_gt) + '\n')
                total += 1
                for k in correct.keys():
                    if correct[k]:
                        score[k] += 1
            bstate_gt = []
            bstate_pred = []
            for k in predictions[0]['gts'].keys():
                correct[k] = True
                correct['joint'] = True
        last_idx = idx

        for prediction in predictions:
            if prediction['gts']['ga'] == 0:
                if prediction['predictions']['ga'] != prediction['gts']['ga']:
                    correct['ga'] = False
                    correct['joint'] = False
            else:
                if prediction['gts']['ga'] != prediction['predictions']['ga']:
                    correct['ga'] = False
                    correct['joint'] = False
                if prediction['gts']['ac'] != prediction['predictions']['ac']:
                    correct['ac'] = False
                    correct['joint'] = False
                if slot in open_slots:
                    if (include_open_span or slot != 'open span') and prediction['gts']['os'] != prediction['predictions']['os']:
                        correct['os'] = False
                        correct['joint'] = False
                else:
                    if prediction['gts']['sl'] != prediction['predictions']['sl']:
                        correct['sl'] = False
                        correct['joint'] = False

        for prediction in predictions:
            sample = all_samples[i]
            input_text = all_samples.tokenizer.decode(sample['input_ids'])
            if slot in open_slots:
                if prediction['gts']['ga'] == 1 and (include_open_span or slot != 'open span'):
                    span = np.array(prediction['gts']['os'])
                    value = all_samples.tokenizer.decode(sample['input_ids'].detach().cpu().numpy()[span != 0])
                    act = acts[prediction['gts']['ac']]
                    bstate_gt.append(f'{slot} {value} {act}')
                if prediction['predictions']['ga'] == 1 and (include_open_span or slot != 'open span'):
                    span = np.array(prediction['predictions']['os'])
                    value = all_samples.tokenizer.decode(sample['input_ids'].detach().cpu().numpy()[span != 0])
                    act = acts[prediction['predictions']['ac']]
                    bstate_pred.append(f'{slot} {value} {act}'.replace('  ', ' '))
            else:
                # print(slot_values[slot])
                if prediction['gts']['ga'] == 1:
                    # print('here')
                    value = slot_values[slot][prediction['gts']['sl']]
                    act = acts[prediction['gts']['ac']]
                    bstate_gt.append(f'{slot} {value} {act}')
                    # print(bstate_gt)
                    # input()
                if prediction['predictions']['ga'] == 1:
                    # print('here11')
                    value = slot_values[slot][prediction['predictions']['sl']]
                    act = acts[prediction['predictions']['ac']]
                    bstate_pred.append(f'{slot} {value} {act}')
                    # print(bstate_pred)
                    # input()

    for k in score.keys():
        print(k, score[k] / total)
