import os.path
import sys
import math
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_fscore_support
import torch
from torch import nn
import random
from torchvision import transforms
import time
import torch.distributed as dist
import datetime
from collections import defaultdict, deque
import h5py
import wandb
from sklearn.model_selection import StratifiedKFold

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, specificity, F1, auc

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


def softmax_one(x, dim=-1):
    # Calculate the exponentials of the input tensor
    exp_x = torch.exp(x)

    # Add 1 to the denominator of the softmax function
    denominator = exp_x.sum(dim=dim, keepdim=True) + 1

    # Calculate the modified softmax
    modified_softmax = exp_x / denominator

    return modified_softmax


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class RandomRotate90:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

def set_seed(seed):
    # Set random seed for PyTorch
    torch.manual_seed(seed)

    # Set random seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for random module
    random.seed(seed)

    # Set random seed for CuDNN if available
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.warmup_epoch:
        lr = cfg.lr * epoch / cfg.warmup_epoch
    else:
        lr = cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg.warmup_epoch) / (cfg.train_epoch - cfg.warmup_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate_StepLR(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.train_epoch // 2:
        lr = cfg.lr
    else:
        lr = cfg.lr * 0.1
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def shuffle_batch(x, shuffle_idx=None):
    """ shuffles each instance in batch the same way """
    
    if not torch.is_tensor(shuffle_idx):
        seq_len = x.shape[1]
        shuffle_idx = torch.randperm(seq_len)
    x = x[:, shuffle_idx]
    
    return x, shuffle_idx

def shuffle_instance(x, axis, shuffle_idx=None):
    """ shuffles each instance in batch in a different way """

    if not torch.is_tensor(shuffle_idx):
        # get permutation indices
        shuffle_idx = torch.rand(x.shape[:axis+1], device=x.device).argsort(axis)  
    
    idx_expand = shuffle_idx.clone().to(x.device)
    for _ in range(x.ndim-axis-1):
        idx_expand.unsqueeze_(-1)
    # reformat for gather operation
    idx_expand = idx_expand.repeat(*[1 for _ in range(axis+1)], *(x.shape[axis+1:]))  
    
    x = x.gather(axis, idx_expand)

    return x, shuffle_idx

class Logger(nn.Module):
    ''' Stores and computes statistiscs of losses and metrics '''

    def __init__(self, task_dict):
        super().__init__()

        self.task_dict = task_dict
        self.losses_it = defaultdict(list)
        self.losses_epoch = defaultdict(list)
        self.y_preds = defaultdict(list)
        self.y_trues = defaultdict(list)
        self.metrics = defaultdict(list)

    def update(self, next_loss, next_y_pred, next_y_true):

        for task in self.task_dict.values():
            t, t_metr = task['name'], task['metric']
            self.losses_it[t].append(next_loss[t])
            
            if t_metr == 'accuracy':
                y_pred = np.argmax(next_y_pred[t], axis=-1)
            elif t_metr in ['multilabel_accuracy', 'auc']:
                y_pred = next_y_pred[t].tolist()
            self.y_preds[t].extend(y_pred)
            
            self.y_trues[t].extend(next_y_true[t])

    def compute_metric(self):

        for task in self.task_dict.values():
            t = task['name']
            losses = self.losses_it[t]
            self.losses_epoch[t].append(np.mean(losses))

            current_metric = task['metric']
            if current_metric == 'accuracy':
                metric = accuracy_score(self.y_trues[t], self.y_preds[t])
                self.metrics[t].append(metric)
            elif current_metric == 'multilabel_accuracy':
                y_pred = np.array(self.y_preds[t])
                y_true = np.array(self.y_trues[t])
                
                y_pred = np.where(y_pred >= 0.5, 1., 0.)
                correct = np.all(y_pred == y_true, axis=-1).sum()
                total = y_pred.shape[0]
                
                self.metrics[t].append(correct / total)
            elif current_metric == 'auc':
                y_pred = np.array(self.y_preds[t])
                y_true = np.array(self.y_trues[t])
                auc = roc_auc_score(y_true, y_pred)
                self.metrics[t].append(auc)

            # reset per iteration losses, preds and labels
            self.losses_it[t] = []
            self.y_preds[t] = []
            self.y_trues[t] = []


    def print_stats(self, epoch, train, **kwargs):

        print_str = 'Train' if train else 'Test'
        print_str +=  " Epoch: {} \n".format(epoch+1)

        avg_loss = 0
        for task in self.task_dict.values():
            t = task['name']
            metric_name = task['metric']
            mean_loss = self.losses_epoch[t][epoch]
            metric = self.metrics[t][epoch]
           
            avg_loss += mean_loss
 
            print_str += "task: {}, mean loss: {:.5f}, {}: {:.5f}, ".format(t, mean_loss, metric_name, metric)

        avg_loss /= len(self.task_dict.values())
        print_str += "avg. loss over tasks: {:.5f}".format(avg_loss)

        for k, v in kwargs.items():
            print_str += ", {}: {}".format(k, v)
        print_str += "\n"

        print(print_str)

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def save_model(conf, epoch, model, optimizer, is_best=False, is_last=False):
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': conf,
    }

    checkpoint_path = os.path.join(conf.ckpt_dir, 'checkpoint-%s.pth' % epoch)

    # record the checkpoint with best validation accuracy
    if is_best:
        checkpoint_path = os.path.join(conf.ckpt_dir, 'checkpoint-best.pth')

    if is_last:
        checkpoint_path = os.path.join(conf.ckpt_dir, 'checkpoint-last.pth')

    torch.save(to_save, checkpoint_path)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class Wandb_Writer(object):

    def __init__(self, project_name='wsi_classification', group_name='baseline', mode='online', name=0):
        self.wandb = wandb.init(project=project_name, group=group_name, entity="dazhangyu123", save_code=True, mode=mode, name='seed%d'%name)

    def log(self, var_name, var, commit=True):
        self.wandb.log({var_name: var}, commit=commit)

    def summary(self, var_name, var):
        self.wandb.run.summary[var_name] = var

def build_transform(transform_list):
    transform = []
    if  'centercrop' in transform_list:
        transform.append(transforms.CenterCrop((1024, 1024)))
    if 'resize' in transform_list:
        transform.append(transforms.Resize((224, 224)))
    if 'flip' in transform_list:
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.RandomVerticalFlip())
    if 'rotate' in transform_list:
        transform.append(RandomRotate90([0, 90, 180, 270]))
    if 'colorJitter' in transform_list:
        transform.append(transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ]))
    if 'totensor' in transform_list:
        transform.append(transforms.ToTensor())
    if 'normalize' in transform_list:
        transform.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return transforms.Compose(transform)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def group_shuffle(x,group=0):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))
    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps,torch.tensor([-1 for i in range(_pad)])])
        ps = ps.view(group,-1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps>=0].view(p)
    else:
        idx = torch.randperm(p)
    return x[:,idx.long()]

def patch_shuffle(x, group=0, g_idx=None, return_g_idx=False):
    b, p, n = x.size()
    ps = torch.tensor(list(range(p)))

    # padding
    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    if group > H or group <= 0:
        return group_shuffle(x, group)
    _n = -H % group
    H, W = H + _n, W + _n
    add_length = H * W - p
    # print(add_length)
    ps = torch.cat([ps, torch.tensor([-1 for i in range(add_length)])])
    # patchify
    ps = ps.reshape(shape=(group, H // group, group, W // group))
    ps = torch.einsum('hpwq->hwpq', ps)
    ps = ps.reshape(shape=(group ** 2, H // group, W // group))
    # shuffle
    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]
    # unpatchify
    ps = ps.reshape(shape=(group, group, H // group, W // group))
    ps = torch.einsum('hwpq->hpwq', ps)
    ps = ps.reshape(shape=(H, W))
    idx = ps[ps >= 0].view(p)

    if return_g_idx:
        return x[:, idx.long()], g_idx
    else:
        return x[:, idx.long()]

def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label >= threshold_optimal] = 1
    this_class_label[this_class_label < threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = accuracy_score(bag_labels, bag_predictions)
    # accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore


@torch.no_grad()
def ema_update(model,targ_model,mm=0.9999):
    r"""Performs a momentum update of the target network's weights.
    Args:
        mm (float): Momentum used in moving average update.
    """
    assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) # mm*k +(1-mm)*q


def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=True):
    """
    dataset split: split the full_list randomly into two sublist (val-set and train-set) based on the ratio
    :param full_list:
    :param ratio:
    :param shuffle:
    """
    # select the val-set based on the label ratio
    if label_balance_val and label is not None:
        _label = label[full_list]
        _label_uni = np.unique(_label)
        sublist_1 = []
        sublist_2 = []

        for _l in _label_uni:
            _list = full_list[_label == _l]
            n_total = len(_list)
            offset = int(n_total * ratio)
            if shuffle:
                random.shuffle(_list)
            sublist_1.extend(_list[:offset])
            sublist_2.extend(_list[offset:])
    else:
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        val_set = full_list[:offset]
        train_set = full_list[offset:]

    return val_set, train_set

def get_kflod(k, patients_array, labels_array, val_ratio=False, label_balance_val=True):
    if k > 1:
        skf = StratifiedKFold(n_splits=k)
    else:
        raise NotImplementedError
    train_patients_list = []
    train_labels_list = []
    test_patients_list = []
    test_labels_list = []
    val_patients_list = []
    val_labels_list = []
    for train_index, test_index in skf.split(patients_array, labels_array):
        if val_ratio != 0.:
            val_index, train_index = data_split(train_index, val_ratio, True, labels_array, label_balance_val)
            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val, y_val = [], []
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)

    # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
    return np.array(train_patients_list, dtype=object), np.array(train_labels_list, dtype=object), np.array(
        test_patients_list, dtype=object), np.array(test_labels_list, dtype=object), np.array(val_patients_list,
                                                                                              dtype=object), np.array(
        val_labels_list, dtype=object)


