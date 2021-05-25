import torch
import torch.nn as nn
from base_model import *
from transformers import AutoModel, AutoConfig, AdamW
from collections import defaultdict
from dataloader import slot_values_keys, slot_tokens, acts
from tqdm import tqdm


class DST(BaseModule):
    def __init__(
        self,
        transformer_model,
        config=None,
        lr=1e-5,
        dropout_transformer=0.1,
        dropout=0.2,
        cuda=True,
        warmup_ratio=0.1,
        num_training_steps=1000,
        device_idxs=(),
        mixed_precision=False,
        # efficientnet_batchsize=32,
        pad_value=0
    ):
        super().__init__(cuda=cuda,
                         warmup_ratio=warmup_ratio,
                         num_training_steps=num_training_steps,
                         device_idxs=device_idxs,
                         mixed_precision=mixed_precision)

        self.config = config or AutoConfig.from_pretrained(transformer_model, hidden_dropout_prob=dropout_transformer, attention_probs_dropout_prob=dropout_transformer)
        self.transformer_fe = AutoModel.from_pretrained(transformer_model, config=self.config)
        for param in self.transformer_fe.parameters():
            param.requires_grad = False
        self.transformer_fe.eval()
        self.transformer = AutoModel.from_pretrained(transformer_model, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier_gate = nn.Linear(self.config.hidden_size, 2)
        self.classifier_span = nn.Linear(self.config.hidden_size, 3)
        self.classifier_action = nn.Linear(self.config.hidden_size, len(acts))

        self.cross_entropy = nn.CrossEntropyLoss()

        self.reset()

        if self.cuda and len(self.devices) > 1:
            self.transformer_fe = nn.DataParallel(self.transformer_fe,
                                                  device_ids=self.devices,
                                                  output_device=self.model_device)
            self.transformer = nn.DataParallel(self.transformer,
                                               device_ids=self.devices,
                                               output_device=self.model_device)

        if self.mixed_precision:
            self.transformer.forward = insert_autocast(self.transformer.forward)

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        self.scheduler = self.linear_scheduler(self.optimizer)

        self.pad_value = pad_value
        self.candidate_value_cache = {}

        self.main_losses = {'os', 'ga', 'ac', 'sl'}

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.transformer_fe.eval()

    def forward(
        self,
        input_ids,
        slot,
        token_type_ids=None,
        attention_mask=None,
        cache=True
    ):
        input_ids = input_ids.to(self.model_device)
        token_type_ids = token_type_ids.to(self.model_device)
        attention_mask = attention_mask.to(self.model_device)
        outputs = self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs[0])
        pooled_output = self.dropout(outputs[1])

        span = self.classifier_span(hidden_states)
        gate = self.classifier_gate(pooled_output)
        action = self.classifier_action(pooled_output)

        cosine_matching = []
        for i in range(len(slot)):
            candidate_tokens = slot_tokens[slot_values_keys[slot[i].item()]]
            cosine_matching.append(torch.zeros((len(candidate_tokens),)).to(self.model_device))
            for j, candidate_token in enumerate(candidate_tokens):
                tuple_token = tuple(candidate_token.squeeze(0).numpy())
                if cache and tuple_token in self.candidate_value_cache:
                    candidate_output = self.candidate_value_cache[tuple_token]
                else:
                    candidate_token = candidate_token.to(self.model_device)
                    token_type_ids_curr = torch.ones_like(candidate_token)
                    token_type_ids_curr[..., 0] = 0
                    with torch.no_grad():
                        candidate_output = self.transformer_fe(candidate_token, token_type_ids=token_type_ids_curr)[1]
                    if cache:
                        self.candidate_value_cache[tuple_token] = candidate_output
                cosine_matching[i][j] = pooled_output[i].unsqueeze(0).mm(candidate_output.t()) / (pooled_output[i].norm() * candidate_output.norm())

        return {
            'span': span,
            'gate': gate,
            'action': action,
            'slot': cosine_matching
        }

    def reset(self):
        self.loss, self.loss_os, self.loss_ga, self.loss_ac, self.loss_sl, self.iter = 0, 0, 0, 0, 0, 1

    def print_loss(self):
        loss_avg = self.loss / self.iter
        loss_os = self.loss_os / self.iter
        loss_ga = self.loss_ga / self.iter
        loss_ac = self.loss_ac / self.iter
        loss_sl = self.loss_sl / self.iter
        self.iter += 1
        return 'L:{:.2f} LO:{:.3f} LG:{:.3f} LA:{:.3f} LS: {:.3f}'.format(loss_avg, loss_os, loss_ga, loss_ac, loss_sl)

    def prepare(self, batch):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']
        span = batch['span']
        gate = batch['gate']
        action = batch['action']
        slot = batch['slot']
        slot_value = batch['slot value']
        ids = batch['id']
        input_ids_len = batch['input_ids_len']

        span[span == -1] = self.cross_entropy.ignore_index
        for i, s in enumerate(span):
            if s[s != self.cross_entropy.ignore_index].sum() == 0:
                span[i] = self.cross_entropy.ignore_index
        action[action == -1] = self.cross_entropy.ignore_index

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'slot': slot
        }, {
            'id': ids,
            'input_ids_len': input_ids_len,
            'span': span,
            'gate': gate,
            'action': action,
            'slot_value': slot_value
        }

    def accumulate_loss(self, outputs, extras):
        span_pred = outputs['span']
        gate_pred = outputs['gate']
        action_pred = outputs['action']
        slot_pred = outputs['slot']
        span_gt = extras['span'].to(self.model_device)
        gate_gt = extras['gate'].to(self.model_device)
        action_gt = extras['action'].to(self.model_device)
        slot_gt = extras['slot_value']

        batch_loss_ga = self.cross_entropy(gate_pred, gate_gt)
        batch_loss_os = self.cross_entropy(span_pred.view(-1, span_pred.shape[-1]), span_gt.view(-1))
        batch_loss_ac = self.cross_entropy(action_pred, action_gt)
        batch_loss_sl = 0
        fixed_slot_sample_count = 0
        for i, slot_pd in enumerate(slot_pred):
            value_idx = slot_gt[i].item()
            if value_idx != -1:
                fixed_slot = slot_pd.detach().clone()
                fixed_slot[value_idx] = -1e7
                loss_fixed_slot = 0.2 - slot_pd[value_idx] + slot_pd[fixed_slot.argmax()]
                if loss_fixed_slot.item() > 0:
                    batch_loss_sl += loss_fixed_slot
                fixed_slot_sample_count += 1

        loss = 2*batch_loss_ga + 10*batch_loss_os + batch_loss_ac
        if fixed_slot_sample_count and bool(batch_loss_sl != 0):
            batch_loss_sl /= fixed_slot_sample_count
            loss += batch_loss_sl
            self.loss_sl += batch_loss_sl.item()
        self.loss_ga += batch_loss_ga.item()
        self.loss_os += batch_loss_os.item()
        self.loss_ac += batch_loss_ac.item()

        self.loss_grad = loss
        self.loss += loss.data

    def make_results(self, outputs, extras):
        span_pred = outputs['span'].detach().clone().argmax(dim=-1)
        gate_pred = outputs['gate'].detach().clone().argmax(dim=-1)
        action_pred = outputs['action'].detach().clone().argmax(dim=-1)
        slot_pred = outputs['slot']
        span_gt = extras['span']
        gate_gt = extras['gate']
        action_gt = extras['action']
        slot_gt = extras['slot_value']
        ids = extras['id']
        input_ids_len = extras['input_ids_len']

        results = defaultdict(list)

        for i, span_gt_out in enumerate(span_gt):
            id2write = ids[i].item() if ids[i].nelement() == 1 else str(list(ids[i].numpy()))
            span_gt_out[span_gt_out == self.cross_entropy.ignore_index] = 0
            span_gt_out = span_gt_out[:input_ids_len[i]].tolist()
            span_pred_out = span_pred[i][:len(span_gt_out)].tolist()
            gate_pred_out = gate_pred[i].tolist()
            gate_gt_out = gate_gt[i].tolist()
            action_gt_out = action_gt[i].item()
            action_pred_out = action_pred[i].item()
            slot_gt_out = slot_gt[i].item()
            if len(slot_pred[i].detach().clone()) == 0:
                slot_pred_out = -1
            else:
                slot_pred_out = slot_pred[i].detach().clone().argmax().item()

            predictions = {
                'ga': gate_pred_out,
                'os': span_pred_out,
                'ac': action_pred_out,
                'sl': slot_pred_out
            }
            gts = {
                'ga': gate_gt_out,
                'os': span_gt_out,
                'ac': action_gt_out,
                'sl': slot_gt_out
            }
            results[id2write].append({
                'predictions': predictions,
                'gts': gts
            })

        return results
