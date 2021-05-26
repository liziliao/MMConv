import torch
import torch.nn as nn
from base_model import *
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR
import timm
from timm.optim import RMSpropTF
from scheduler import StepLRScheduler
from collections import defaultdict


class EfficientNet(BaseModule):
    def __init__(
        self,
        baseline_model,
        pretrained=True,
        num_classes=None,
        lr=6.25e-5,
        dropout=0.2,
        drop_connect=0.2,
        cuda=True,
        warmup_ratio=0.1,
        num_training_steps=1000,
        gamma=0.97,
        device_idxs=(),
        mixed_precision=False
    ):
        super().__init__(cuda=cuda,
                         warmup_ratio=warmup_ratio,
                         num_training_steps=num_training_steps,
                         device_idxs=device_idxs,
                         mixed_precision=mixed_precision)

        self.baseline_model = timm.create_model(
            baseline_model,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_connect_rate=drop_connect,
            drop_path_rate=drop_connect
        )

        self.cross_entropy = nn.CrossEntropyLoss()

        self.reset()

        if self.cuda and len(self.devices) > 1:
            self.baseline_model = nn.DataParallel(self.baseline_model,
                                                 device_ids=self.devices,
                                                 output_device=self.model_device)

        if self.mixed_precision:
            self.baseline_model.forward = insert_autocast(self.baseline_model.forward)

        if True:
            self.optimizer = RMSpropTF(self.parameters(), alpha=0.9, momentum=0.9, weight_decay=1e-5, eps=1e-3, lr=lr)
            self.scheduler = StepLRScheduler(self.optimizer,
                                             decay_t=self.num_training_steps * 2.4 / 450,
                                             decay_rate=gamma,
                                             warmup_lr_init=1e-6,
                                             warmup_t=self.num_training_steps * 3 / 450,
                                             noise_range_t=None,
                                             noise_pct=0.67,
                                             noise_std=1,
                                             noise_seed=42)
        else:
            self.optimizer = AdamW(self.parameters(), lr=lr)
            self.scheduler = StepLR(self.optimizer, step_size=int(num_training_steps * 2.4 / 450), gamma=gamma)

        self.main_losses = {'im'}

    def forward(self, images):
        logits = self.baseline_model(images)
        return {
            'im': logits
        }

    def reset(self):
        self.loss, self.loss_im, self.iter = 0, 0, 1

    def print_loss(self):
        loss_avg = self.loss / self.iter
        loss_im = self.loss_im / self.iter
        self.iter += 1
        return 'L:{:.2f} LIM:{:.3f}'.format(loss_avg, loss_im)

    def prepare(self, batch):
        images = batch['image']
        labels = batch['label']
        ids = batch['id']

        return {
            'images': images
        }, {
            'id': ids,
            'labels': labels
        }

    def accumulate_loss(self, outputs, extras):
        logits = outputs['im']
        labels = extras['labels']

        batch_loss_im = self.cross_entropy(logits, labels.to(self.model_device))

        loss = batch_loss_im
        self.loss_im += batch_loss_im.item()

        self.loss_grad = loss
        self.loss += loss.data

    def make_results(self, outputs, extras):
        im = outputs['im'].detach().cpu().clone()
        labels = extras['labels'].detach().cpu().clone()
        ids = extras['id']

        im = im.argmax(-1).long()

        results = defaultdict(list)

        for i, label in enumerate(labels):
            pred = im[i].tolist()
            gt = label.tolist()

            results[ids[i].item()].append({
                'predictions': {
                    'im': pred
                },
                'gts': {
                    'im': gt
                }
            })

        return results
