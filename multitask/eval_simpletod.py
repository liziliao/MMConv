import torch
torch.backends.cudnn.benchmark = True
from utils.model_utils import get_device_ids
from transformers import GPT2Config, GPT2LMHeadModel
from dataloader_simpletod import MMDialDataset
from torch.utils.data import DataLoader
from utils.model_utils import CustomPaddingTensorCollator
from torch.nn import DataParallel
import sys
import os
from collections import defaultdict
from tqdm import tqdm
from utils.json_utils import save, load
import re

informable_slots = {'wheelchair accessible', 'reservations', 'restroom',
                    'smoking', 'credit cards', 'outdoor seating', 'parking',
                    'music', 'wi-fi', 'dining options', 'drinks', 'venuescore',
                    'menus', 'price', 'venueneigh'}
requestable_slots = {'venuename', 'telephone', 'venueaddress'}

token_matcher = re.compile(r'<\|[a-zA-Z]+\|>')


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


transformer_model = sys.argv[1]

device_ids = get_device_ids(cuda=True)
print('Using devices: {}'.format(device_ids))

BATCH_SIZE_EVAL = int(sys.argv[2])
print('Batch size for evaluation: {}'.format(BATCH_SIZE_EVAL))

workers_eval = max(min(8, BATCH_SIZE_EVAL >> 3), 4)


paths = ['./resources/test.simpletod']   
test = MMDialDataset.create_data(paths, transformer_model, split=(1,), shuffle=False)

first = {
    'context_input_ids': True,
    'context_attention_mask': True
}
key2pad_id = {
    'context_input_ids': test.tokenizer.eos_token_id,
    'labels': test.tokenizer.eos_token_id
}
collator = CustomPaddingTensorCollator(first=first, key2pad_id=key2pad_id)
dataloader_test = DataLoader(test, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False, collate_fn=collator)

checkpoint = sys.argv[3]
config = GPT2Config.from_pretrained(os.path.join(checkpoint, 'config.json'))
model = GPT2LMHeadModel.from_pretrained(os.path.join(checkpoint, 'pytorch_model.bin'), config=config)
model.eval()
model_device = device_ids[0]

if len(device_ids) > 1:
    model = DataParallel(model, device_ids=device_ids, output_device=device_ids[-1])
model.to(f'cuda:{model_device}')



def make_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def slot_in_slots(slot, slots):
    if not slot.strip():
        return False
    slot_split = slot.split()
    return slot_split[0] in slots or ' '.join(slot_split[:2]) in slots


def get_belief(belief, slots=None):
    return [x for x in belief.split(', ') if slots is None or slot_in_slots(x, slots)]


def shift_past(past, shift=1):
    return tuple(p[:, :, :, shift:, :] for p in past)


max_len = 800
with torch.no_grad():
    print("STARTING EVALUATION")
    all_predictions = defaultdict(list)
    pbar = tqdm(enumerate(dataloader_test), total=len(dataloader_test))
    end_tokens = [1279, 91, 437, 1659, 26209, 91, 29] # <|endofresponse|>
    end_tokens = torch.tensor(end_tokens).to(model_device).view(-1)
    for i, batch in pbar:
        input_ids = batch['context_input_ids'].long().to(model_device)
        attention_mask = batch['context_attention_mask'].long().to(model_device)
        labels = batch['labels'].long()
        labels_len = batch['labels_len'].long()
        ids = batch['id'].long()

        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        generation_mask = torch.full((input_ids.shape[0], end_tokens.shape[0]), -1).to(model_device)
        next_tokens = input_ids
        predictions = []
        past = None
        for i in range(max_len):
            position_ids = make_position_ids(attention_mask)
            if past is not None:
                position_ids = position_ids[:, -1]
            outputs = model(next_tokens, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past, use_cache=True, return_dict=True)
            logits = outputs['logits'][:, -1].unsqueeze(1)
            past = outputs['past_key_values']
            next_tokens = logits.argmax(dim=-1)
            if i < generation_mask.shape[1]:
                diff_mask = torch.tensor(1).to(model_device)
                generation_mask[:, i] = next_tokens.view(-1)
            else:
                diff_mask = (generation_mask != end_tokens).any(dim=1)
                if generation_mask[diff_mask].nelement() == 0:
                    break
                generation_mask[diff_mask] = generation_mask[diff_mask].roll(-1, 1)
                generation_mask[diff_mask, -1] = next_tokens.view(-1)[diff_mask]
            if attention_mask.shape[1] < 1024:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(model_device) * diff_mask.view(-1, 1)], dim=1)
            else:
                attention_mask = attention_mask.roll(-1, 1)
                attention_mask[:, -1] = torch.ones(attention_mask.shape[0]).to(model_device) * diff_mask.view(-1)
                past = shift_past(past)
            predicted_tokens = next_tokens.detach()
            predicted_tokens[attention_mask[:, -1] == 0] = test.tokenizer.eos_token_id
            predictions.append(predicted_tokens)
        predictions = torch.cat(predictions, dim=1)

        for j, gt in enumerate(labels):
            gt = gt[:labels_len[j]].tolist()
            pred = predictions[j][predictions[j] != test.tokenizer.eos_token_id].tolist()
            gt_text = test.tokenizer.decode(gt)
            pred_text = test.tokenizer.decode(pred)
            all_predictions[ids[j].item()].append({
                'response_prediction': MMDialDataset.extract(pred_text, '<|response|>', keep_tokens=True),
                'response_gt': MMDialDataset.extract(gt_text, '<|response|>', keep_tokens=True),
                'belief_prediction': extract(pred_text, '<|belief|>')[0],
                'belief_gt': extract(gt_text, '<|belief|>')[0],
                'action_prediction': extract(pred_text, '<|action|>')[0],
                'action_gt': extract(gt_text, '<|action|>')[0]
            })

    cpn = checkpoint[checkpoint.rindex('/') + 1:]
    save(all_predictions, f'all_prediction_mmdial_{cpn}.json')

    cpn = checkpoint[checkpoint.rindex('/') + 1:]
    all_predictions = load(f'all_prediction_mmdial_{cpn}.json')
    

    score_belief = 0
    score_action = 0
    score_inform = 0
    score_request = 0
    total = 0
    for predictions in all_predictions.values():
        for prediction in predictions:
            total += 1
            ## belief_correct is true when all belief states match the groundtruth
            belief_prediction = set(get_belief(prediction['belief_prediction']))
            belief_gt = set(get_belief(prediction['belief_gt']))
            belief_correct = belief_prediction == belief_gt
            
            inform_prediction = set(get_belief(prediction['action_prediction'], informable_slots))
            inform_gt = set(get_belief(prediction['action_gt'], informable_slots))
            inform_correct = inform_prediction == inform_gt
            request_prediction = set(get_belief(prediction['action_prediction'], requestable_slots))
            request_gt = set(get_belief(prediction['action_gt'], requestable_slots))
            request_correct = request_prediction == request_gt
            
            # inform rate is match rate, meaning the venuename matches
 
            if belief_correct:
                score_belief += 1
            if inform_correct:
                score_inform += 1
            if request_correct:
                score_request += 1
            action_prediction = set(get_belief(prediction['action_prediction']))
            action_gt = set(get_belief(prediction['action_gt']))
            action_correct = action_prediction == action_gt
            if action_correct:
                score_action += 1

    # print(f'Bleu 2: {bleu_score_2}\nBleu 4: {bleu_score_4}')
    print(f'Belief acc: {score_belief / total}\nAction acc: {score_action / total}\nInform acc: {score_inform / total}\nRequest acc: {score_request / total}')
