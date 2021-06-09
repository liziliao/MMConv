import torch
from utils.generic_utils import get_empty_cuda_devices
from collections import defaultdict
from tqdm import tqdm
from utils.generic_utils import current_time
from base_model import BaseModule
from utils.json_utils import save
import os


def set_max_len(*datasets, key='response', strict=False):
    max_len = 0
    for dataset in datasets:
        for sample in dataset:
            max_len = max(max_len, len(sample['response'][0]))
    if not strict:
        max_len = round(max_len + 5, -1)
    for dataset in datasets:
        dataset.max_len = max_len


def get_device_ids(cuda=True):
    if cuda:
        print('Using cuda')
        all_devices = get_empty_cuda_devices()
        device_ids = input('Found {} idle device(s): {}\nPlease choose which one(s) to use (default: \'all\'): '.format(
            len(all_devices), all_devices))
        if not device_ids or device_ids.lower() == 'all':
            device_ids = all_devices
        else:
            device_ids = sorted(set(int(i) for i in device_ids.split()))
    else:
        device_ids = ['cpu']
    return device_ids


def collate_dict(batch):
    collated_batch = defaultdict(list)
    for k in batch[0].keys():
        collated_batch[k] = [sample[k] for sample in batch]
    return collated_batch


def collate_tensor(batch):
    collated_batch = defaultdict(torch.Tensor)
    for k in batch[0].keys():
        collated_batch[k] = BaseModule.pad_seq([sample[k] for sample in batch])
        if len(collated_batch[k].shape) > 1:
            collated_batch[k] = collated_batch[k].squeeze(1)
    return collated_batch


class CustomPaddingTensorCollator():
    def __init__(self, key2pad_id={}, first={}, ignored_keys=set(), common_keys=set()):
        self.key2pad_id = key2pad_id
        self.first = first
        self.ignored_keys = ignored_keys
        self.common_keys = common_keys

    def __call__(self, batch):
        collated_batch = defaultdict(torch.Tensor)
        for k in batch[0].keys():
            if k in self.ignored_keys:
                collated_batch[k] = [sample[k] for sample in batch]
            elif k in self.common_keys:
                collated_batch[k] = batch[0][k]
            else:
                collated_batch[k] = BaseModule.pad_seq([sample[k] for sample in batch], val=self.key2pad_id.get(k, 0), first=self.first.get(k, False))
                if len(collated_batch[k].shape) > 1:
                    collated_batch[k] = collated_batch[k].squeeze(1)
        return collated_batch


def main_loop(model, dataloader, epochs=1, batch_iters=1,
              clear_cache_every=100, train=True, cuda=True,
              calc_loss=True, calc_acc=True, model_ema=None,
              save_best=True, acc_thresh=0, model_name='MODEL',
              acc_nchars=1, save_results=False, print_all_acc=True):
    if cuda:
        torch.cuda.empty_cache()
    model.reset()
    model.zero_grad()
    if not train and model_ema is not None:
        model_ema.module.reset()
        model_ema.module.zero_grad()
    scores = defaultdict(int)
    total = defaultdict(int)
    if save_results:
        all_results = defaultdict(list)
    for epoch in range(epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in pbar:
            # Forward
            inputs, extras = model.prepare(batch)
            if not train and model_ema is not None:
                outputs = model_ema.module(**inputs)
            else:
                outputs = model(**inputs)
            if calc_acc:
                curr_total = defaultdict(int)
                results = model.make_results(outputs, extras)
                for id_, result in results.items():
                    if save_results:
                        all_results[id_].extend(result)
                    for sample in result:
                        if sample:
                            curr_total['joint'] += 1
                            joint_correct = True
                            for target in sample['predictions']:
                                curr_total[target] += 1
                                if sample['predictions'][target] == sample['gts'][target]:
                                    scores[target] += 1
                                else:
                                    joint_correct = False
                            if joint_correct:
                                scores['joint'] += 1
                for k in curr_total.keys():
                    total[k] += curr_total[k]
                acc = {'A' + target[:acc_nchars].upper(): BaseModule.ratio(scores[target], total[target]) for target in scores if target in model.main_losses}

            # Total mini batches for loss normalization
            if i + batch_iters > len(dataloader):
                total_mini_batches = (len(dataloader) % batch_iters) or batch_iters
            else:
                total_mini_batches = batch_iters
            if train or calc_loss:
                model.accumulate_loss(outputs, extras)
            if train:
                # Backward
                model.backward(r=1 / total_mini_batches, l2=False)
                if (i + 1) % batch_iters == 0:
                    model.optimize()
                    model.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
            if cuda and (i + 1) % clear_cache_every == 0:
                torch.cuda.empty_cache()
            description = f'LE:{epoch + 1}/{epochs}'
            if calc_loss:
                description += ' ' + model.print_loss()
            #if calc_acc:
            #    description += ' ' + ' '.join([f'{k}:{v}' for k, v in acc.items()])
            pbar.set_description(description)
        if train and len(dataloader) % batch_iters:
            model.optimize()
            model.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        if print_all_acc:
            print({'ACC_' + target.upper(): BaseModule.ratio(scores[target], total[target]) for target in scores})
    if save_results:
        os.makedirs('prediction', exist_ok=True)
        save(all_results, os.path.join('prediction', model_name + ('EMA' if not train and model_ema is not None else '') + '_' + current_time().replace('/', '-') + '_val_acc-' + ' '.join([f'{k}:{v}' for k, v in acc.items()]) + '_val_loss-' + model.print_loss() + '.json'))
    if not train and save_best:
        joint_acc = BaseModule.ratio(scores['joint'], total['joint'])
        if joint_acc >= acc_thresh:
            model.save_model(model_name + '_' + current_time().replace('/', '-') + '_val_acc-' + ' '.join([f'{k}:{v}' for k, v in acc.items()]) + '_val_loss-' + model.print_loss())
            if not train and model_ema is not None:
                model_ema.module.save_model(model_name + 'EMA_' + current_time().replace('/', '-') + '_val_acc-' + ' '.join([f'{k}:{v}' for k, v in acc.items()]) + '_val_loss-' + model.print_loss())


def loop(model, dataloader, epochs=1, batch_iters=1, clear_cache_every=100, train=True, cuda=True, calc_loss=True, calc_acc=True, model_ema=None, save_best=True, acc_thresh=0, model_name='MODEL', acc_nchars=1, save_results=False, print_all_acc=True):
    model.train(train)
    if train:
        print('TRAINING...')
        main_loop(model, dataloader, epochs=epochs, batch_iters=batch_iters, clear_cache_every=clear_cache_every,
                  train=train, cuda=cuda, calc_loss=calc_loss, calc_acc=calc_acc, model_ema=model_ema, save_best=save_best, acc_thresh=acc_thresh, model_name=model_name, acc_nchars=acc_nchars, save_results=save_results, print_all_acc=print_all_acc)
    else:
        print('EVALUATING...')
        with torch.no_grad():
            main_loop(model, dataloader, epochs=epochs, batch_iters=batch_iters, clear_cache_every=clear_cache_every,
                      train=train, cuda=cuda, calc_loss=calc_loss, calc_acc=calc_acc, model_ema=model_ema, save_best=save_best, acc_thresh=acc_thresh, model_name=model_name, acc_nchars=acc_nchars, save_results=save_results, print_all_acc=print_all_acc)
