import torch
from torch.nn.functional import pad
import os
from torch.cuda.amp import  GradScaler
from torch.optim import lr_scheduler
import torch.nn as nn
from collections import deque


# Custom DataParallel class for attribute accessing
class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.module, name)


class AutoCast:
    def __init__(self, func, value=True):
        self.func = func
        self.value = value

    def __call__(self, *args, **kwargs):
        with autocast(self.value):
            return self.func(*args, **kwargs)


def insert_autocast(func, value=True):
    return AutoCast(func, value)


class Func:
    def __init__(self, func, pre=None, post=None):
        self.func = func
        self.pre = pre
        self.post = post

    def __call__(self, *args, **kwargs):
        if self.pre is not None:
            self.pre()
        ret = self.func(*args, **kwargs)
        if self.post is not None:
            self.post()
        return ret


def insert_func(func, pre=None, post=None):
    return Func(func, pre, post)


class MethodGroup:
    def __init__(self, methods):
        self.methods = methods

    def __call__(self, *args, **kwargs):
        return [method(*args, **kwargs) for method in self.methods]


class Group(deque):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            ret = [getattr(element, name) for element in self]
            if ret and callable(ret[0]):
                return MethodGroup(ret)
            return ret


class BaseModule(nn.Module):
    def __init__(self,
                 cuda=True,
                 warmup_ratio=0.1,
                 num_training_steps=1000,
                 device_idxs=(),
                 mixed_precision=False):
        super().__init__()

        # Other parameters
        self.num_warmup_steps = int(warmup_ratio * num_training_steps)
        self.num_training_steps = num_training_steps
        self.cuda = cuda
        if self.cuda:
            self.devices = device_idxs
        else:
            self.devices = ['cpu']
        self.model_device = device_idxs[0]
        self.mixed_precision = mixed_precision

        # Mixed precision training support
        if self.mixed_precision:
            self.scaler = GradScaler()

    def linear_scheduler(self, optimizer, last_epoch=-1):
        return lr_scheduler.LambdaLR(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0, float(self.num_training_steps - current_step) / float(
                max(1, self.num_training_steps - self.num_warmup_steps))
        )

    def backward(self, r=1, l2=False):
        # Loss scaling (can be used for accumulation normalizing)
        self.loss_grad = self.loss_grad * r

        # L2 normalization
        if l2:
            if self.mixed_precision:
                grad_params = torch.autograd.grad(self.scaler.scale(self.loss_grad), self.parameters(),
                                                  create_graph=True)
                inv_scale = 1 / self.scaler.get_scale()
                grad_params = [p * inv_scale for p in grad_params]
            else:
                grad_params = torch.autograd.grad(self.loss_grad, self.parameters(), create_graph=True)
            with autocast(self.mixed_precision):
                grad_norm = 0
                for grad in grad_params:
                    grad_norm += grad.pow(2).sum()
                grad_norm = grad_norm.sqrt()
                self.loss_grad = self.loss_grad + grad_norm

        # Backward
        if self.mixed_precision:
            self.scaler.scale(self.loss_grad).backward()
        else:
            self.loss_grad.backward()

    def optimize(self, clip=True):
        if clip:
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()

    def save_model(self, checkpoint_name, state_dict_only=True):
        dataparallel = self.single_gpu()
        if not os.path.isdir('checkpoint'):
            os.makedirs('checkpoint', exist_ok=True)
        save_path = os.path.join('checkpoint', checkpoint_name + '.th')
        if state_dict_only:
            torch.save(self.state_dict(), save_path)
        else:
            torch.save(self, save_path)
        self.multi_gpus(dataparallel)
        saved_component = 'state dict' if state_dict_only else 'model'
        print(f'Saved {saved_component} to {save_path}')

    def load_model(self, path, is_state_dict=True):
        dataparallel = self.single_gpu()
        state_dict = torch.load(path, map_location='cpu')
        if not is_state_dict:
            state_dict = state_dict.state_dict()
        self.load_state_dict(state_dict)
        self.multi_gpus(dataparallel)
        loaded_component = 'state dict' if is_state_dict else 'model'
        print(f'Loaded {loaded_component} from {path}')

    @classmethod
    def tensor(cls, x):
        try:
            return torch.tensor(x)
        except:
            return torch.stack(x)

    @classmethod
    def get_pad_amount(cls, max_lens, x, first=False):
        zeros = torch.zeros_like(max_lens)
        if first:
            idxs = torch.stack([zeros, max_lens - torch.tensor(x.shape)])
        else:
            idxs = torch.stack([max_lens - torch.tensor(x.shape), zeros])
        return list(idxs.T.reshape(-1).flip(0))

    @classmethod
    def pad_seq(cls, x, val=0, first=False):
        if isinstance(x, torch.Tensor):
            return x
        try:
            return BaseModule.tensor(x)
        except:
            x = [BaseModule.pad_seq(x_, val=val, first=first) for x_ in x]
            max_lens = torch.tensor([max(x_.shape[i] for x_ in x) for i in range(x[0].ndim)])
            return torch.stack([pad(x_, pad=BaseModule.get_pad_amount(max_lens, x_, first), value=val) for x_ in x])

    @classmethod
    def getattr(cls, obj, name, *args, **kwargs):
        if '.' in name:
            split_index = name.index('.')
            return cls.getattr(getattr(obj, name[:split_index]), name[split_index + 1:], *args, **kwargs)
        return getattr(obj, name, *args, **kwargs)

    @classmethod
    def setattr(cls, obj, name, value):
        if '.' in name:
            split_index = name.index('.')
            return cls.setattr(getattr(obj, name[:split_index]), name[split_index + 1:], value)
        return setattr(obj, name, value)

    def single_gpu(self):
        dataparallel = set()
        for name, module in self.named_modules():
            if isinstance(module, nn.DataParallel):
                dataparallel.add(name)
        for name in dataparallel:
            BaseModule.setattr(self, name, BaseModule.getattr(self, name).module)
        return dataparallel

    def multi_gpus(self, modules):
        for name in modules:
            BaseModule.setattr(self, name, DataParallel(BaseModule.getattr(self, name), device_ids=self.devices, output_device=self.model_device))

    @classmethod
    def ratio(cls, x, y, ndigits=3):
        if y == 0:
            return 0
        return round(x / y, ndigits)

    def make_position_ids(self, attention_mask):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids
