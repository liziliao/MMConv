import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from dataloader import TextDataset
from model_dst import DST
from utils.model_utils import loop, get_device_ids, CustomPaddingTensorCollator
import sys
from tqdm import tqdm

transformer_model = 'bert-base-uncased'
input_format = 'dst'
model_name = f'{input_format}_{transformer_model}'
cuda = True
mixed_precision = False
clear_cache_every = 200

device_ids = get_device_ids(cuda=cuda)
print('Using devices: {}'.format(device_ids))

BATCH_SIZE = int(sys.argv[1])
print('Batch size: {}'.format(BATCH_SIZE))
MINI_BATCH_SIZE = min((len(device_ids) << 5) if cuda else 32, BATCH_SIZE)
print('Mini batch size for training: {}'.format(MINI_BATCH_SIZE))
BATCH_SIZE_EVAL = min((len(device_ids) << 6) if cuda else 64, BATCH_SIZE << 1)
print('Batch size for evaluation: {}'.format(BATCH_SIZE_EVAL))

assert BATCH_SIZE % MINI_BATCH_SIZE == 0
batch_iters = BATCH_SIZE // MINI_BATCH_SIZE

workers = max(min(16, MINI_BATCH_SIZE >> 3), 4)
workers_eval = max(min(8, BATCH_SIZE_EVAL >> 3), 4)


train = TextDataset.create_data(f'resources/train.{input_format}', tokenizer_or_transformer_model=transformer_model, split=(1,), shuffle=True, task='dst', data_var_token='<|belief|>', split_data_var='; ')
val = TextDataset.create_data(f'resources/val.{input_format}', tokenizer_or_transformer_model=transformer_model, split=(1,), training=False, shuffle=False, task='dst', data_var_token='<|belief|>', split_data_var='; ', reduce_slots_without_values=False)
test = TextDataset.create_data(f'resources/test.{input_format}', tokenizer_or_transformer_model=transformer_model, split=(1,), training=False, shuffle=False, task='dst', data_var_token='<|belief|>', split_data_var='; ', reduce_slots_without_values=False)

pad_value = train.tokenizer.pad_token_id
if pad_value is None:
    pad_value = train.tokenizer.eos_token_id

key2pad_id = {
    'input_ids': pad_value,
    'span': -1
}

collator= CustomPaddingTensorCollator(key2pad_id=key2pad_id)

dataloader_train = DataLoader(train, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False, collate_fn=collator)
dataloader_val = DataLoader(val, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False, collate_fn=collator)
dataloader_test = DataLoader(test, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False, collate_fn=collator)

lr=1e-4
warmup_epochs = 0.5
epochs = 5
epochs_per_val = 1
epochs_per_test = 1
min_epoch_for_val = 0
min_epoch_for_test = 0
steps = epochs * len(dataloader_train)

model = DST(transformer_model, warmup_ratio=warmup_epochs / epochs, num_training_steps=steps, lr=lr, device_idxs=device_ids, mixed_precision=mixed_precision, cuda=cuda, pad_value=pad_value)


if cuda:
    # if len(device_ids) > 1:
    #     model = DataParallel(model, device_ids=device_ids, output_device=device_ids[-1])
    model.to('cuda:' + str(device_ids[0]))

pbar = tqdm(range(epochs))
model_name += '_latest_rebalanced_low_lr'
for i in pbar:
    pbar.set_description(f'Epoch {i + 1}/{epochs}')
    loop(model, dataloader_train, batch_iters=batch_iters, clear_cache_every=clear_cache_every, train=True, cuda=cuda, model_name=model_name)
    if (i + 1) >= min_epoch_for_val and (i + 1) % epochs_per_val == 0:
        loop(model, dataloader_val, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name=model_name)
    if (i + 1) >= min_epoch_for_test and (i + 1) % epochs_per_test == 0:
        loop(model, dataloader_test, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name=model_name, save_best=False, save_results=True)
       