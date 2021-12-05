import os.path as osp
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from torchvision import transforms
from utils.data_util import CharacterTrajectoriesDataset, pad_collate,\
    SkipTransform, TrimZeros, CumSum, Scale, TimeScalePos, TimeScaleVel, TemporalContext
from models import RNNModel
from configs.config_global import DEVICE, MAP_LOC


def grad_clipping(model, max_norm, printing=False):
    p_req_grad = [p for p in model.parameters() if p.requires_grad]

    if printing:
        grad_before = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_before += param_norm.item() ** 2
        grad_before = grad_before ** (1. / 2)

    clip_grad_norm_(p_req_grad, max_norm)

    if printing:
        grad_after = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_after += param_norm.item() ** 2
        grad_after = grad_after ** (1. / 2)

        if grad_before > grad_after:
            print("clipped")
            print("before: ", grad_before)
            print("after: ", grad_after)


def data_init(mode, use_velocity, t_scale, batch_s, context, context_w):
    if mode == 'train':
        mode_flag = True
    elif mode == 'test':
        mode_flag = False
    else:
        raise NotImplementedError

    transforms_list = [TrimZeros(), SkipTransform(skip_num=2)]
    # initialize dataset and data transforms
    if use_velocity:
        # classification based on velocity trajectory
        transforms_list += [TimeScaleVel(t_scale)]
    else:
        # classification based on position trajectory
        transforms_list += [CumSum(), Scale(scale=0.1), TimeScalePos(t_scale)]

    if context is not None:
        transforms_list += [TemporalContext(context, context_w, t_scale)]

    trans = transforms.Compose(transforms_list)
    data_set = CharacterTrajectoriesDataset(large_split=mode_flag, transform=trans)
    data_loader = DataLoader(data_set, batch_size=batch_s, shuffle=mode_flag,
                             collate_fn=pad_collate, drop_last=True)

    return data_loader


def model_init(mode, model_type, input_size, hidden_size, save_path=None):
    model = RNNModel(input_size=input_size, hidden_size=hidden_size,
                     output_size=20, rnn_type=model_type)
    model = model.to(DEVICE)

    if mode == 'train':
        assert save_path is None
    elif mode == 'test':
        assert save_path is not None
        # load model from saved ones
        model_path = osp.join(save_path, 'net_best.pth')
        state_dict = torch.load(model_path, map_location=MAP_LOC)
        model.load_state_dict(state_dict, strict=True)
        print("successfully loaded model:\n" + model_path)
    else:
        raise NotImplementedError

    return model
