import torch
torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from dataloader import ImageDataset, img_classes
from model_en import EfficientNet
from timm.utils import ModelEmaV2
from utils.model_utils import loop, get_device_ids, CustomPaddingTensorCollator
import sys
from tqdm import tqdm

image_size_map = {
    'tf_efficientnet_b0_ns': 224,
    'tf_efficientnet_b1_ns': 240,
    'tf_efficientnet_b2_ns': 260,
    'tf_efficientnet_b3_ns': 300,
    'tf_efficientnet_b4_ns': 380,
    'tf_efficientnet_b5_ns': 456,
    'tf_efficientnet_b6_ns': 528,
    'tf_efficientnet_b7_ns': 600,
    'tf_efficientnet_b8_ns': 672,
    'tf_efficientnet_l2_ns_475': 475,
    'tf_efficientnet_l2_ns': 800
}

baseline_model = 'tf_efficientnet_b4_ns'
image_size = image_size_map[baseline_model]
auto_augment = 'rand-m9-mstd0.5'
input_format = 'image'
model_name = f'{input_format}_{baseline_model}'
cuda = True
mixed_precision = False
clear_cache_every = 200

device_ids = get_device_ids(cuda=cuda)
print('Using devices: {}'.format(device_ids))

BATCH_SIZE = int(sys.argv[1])
print('Batch size: {}'.format(BATCH_SIZE))
MINI_BATCH_SIZE = min((len(device_ids) << 8) if cuda else 512, BATCH_SIZE)
print('Mini batch size for training: {}'.format(MINI_BATCH_SIZE))
BATCH_SIZE_EVAL = min((len(device_ids) << 9) if cuda else 1024, BATCH_SIZE << 1)
print('Batch size for evaluation: {}'.format(BATCH_SIZE_EVAL))

assert BATCH_SIZE % MINI_BATCH_SIZE == 0
batch_iters = BATCH_SIZE // MINI_BATCH_SIZE

workers = max(min(16, MINI_BATCH_SIZE >> 3), 4)
workers_eval = max(min(8, BATCH_SIZE_EVAL >> 3), 4)

# all = ImageDataset.create_data([f'resources/train.{input_format}', f'resources/val.{input_format}', f'resources/test.{input_format}'], image_size=image_size, auto_augment=auto_augment, split=(1,), shuffle=False)
train = ImageDataset.create_data(f'resources/train.{input_format}', image_size=image_size, auto_augment=auto_augment, split=(1,), shuffle=True)
val = ImageDataset.create_data(f'resources/val.{input_format}', image_size=image_size, auto_augment=auto_augment, split=(1,), training=False, shuffle=False)
test = ImageDataset.create_data(f'resources/test.{input_format}', image_size=image_size, auto_augment=auto_augment, split=(1,), training=False, shuffle=False)

dataloader_train = DataLoader(train, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True, drop_last=False)
dataloader_val = DataLoader(val, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False)
dataloader_test = DataLoader(test, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=workers_eval, pin_memory=True, drop_last=False)

epochs = 25
epochs_per_val = 1
epochs_per_test = 1
min_epoch_for_val = 0
min_epoch_for_test = 0
steps = epochs * len(dataloader_train)

model = EfficientNet(baseline_model, num_classes=len(img_classes), num_training_steps=steps, lr=4e-3, device_idxs=device_ids, mixed_precision=mixed_precision, cuda=cuda)

if cuda:
    # if len(device_ids) > 1:
    #     model = DataParallel(model, device_ids=device_ids, output_device=device_ids[-1])
    model.to('cuda:' + str(device_ids[0]))
model_ema = ModelEmaV2(model, decay=0.9999, device=model.model_device)

pbar = tqdm(range(epochs))
for i in pbar:
    pbar.set_description(f'Epoch {i + 1}/{epochs}')
    loop(model, dataloader_train, batch_iters=batch_iters, clear_cache_every=clear_cache_every, train=True, cuda=cuda, model_name=model_name, model_ema=model_ema)
    if (i + 1) >= min_epoch_for_val and (i + 1) % epochs_per_val == 0:
        loop(model, dataloader_val, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name=model_name, model_ema=model_ema)
    if (i + 1) >= min_epoch_for_test and (i + 1) % epochs_per_test == 0:
        loop(model, dataloader_test, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name=model_name, model_ema=model_ema, save_best=False, save_results=True)
        # loop(model, dataloader_test, batch_iters=1, clear_cache_every=clear_cache_every, train=False, cuda=cuda, model_name=model_name, save_best=True, save_results=True)
