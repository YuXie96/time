import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from data_util import CharacterTrajectoriesDataset, pad_collate
from models import RNNModel


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


if __name__ == '__main__':
    readout_steps = 1
    num_classes = 20
    # include_targets = list(range(num_classes))

    train_data = CharacterTrajectoriesDataset(train=False)
    test_data = CharacterTrajectoriesDataset()

    # train_data = get_subset(train_data, include_targets)
    # test_data = get_subset(test_data, include_targets)

    batch_s = 20
    train_loader = DataLoader(train_data, batch_size=batch_s, collate_fn=pad_collate, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_s, shuffle=False, collate_fn=pad_collate, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    model = RNNModel(3, 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters())

    for ep in range(100):
        for data, label in train_loader:
            loss = 0.0
            hid = model.init_hidden(batch_s)
            for t_ in range(data.shape[0]):
                output, hid = model(data[t_], hid)

                if t_ >= data.shape[0] - readout_steps:
                    loss += criterion(output, label)
            optimizer.zero_grad()
            loss /= readout_steps
            grad_clipping(model, 1.0)
            loss.backward()
            optimizer.step()

        print(loss.item())
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in test_loader:
                hid = model.init_hidden(batch_s)
                for t_ in range(data.shape[0]):
                    output, hid = model(data[t_], hid)
                    if t_ >= data.shape[0] - readout_steps:
                        _, predicted = torch.max(output.data, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()

        print('Accuracy of the network on the test set: %d %%' % (
                100 * correct / total))

    # prepare to count predictions for each class
    classes = list(range(num_classes))
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data, label in test_loader:
            hid = model.init_hidden(batch_s)
            for t_ in range(data.shape[0]):
                output, hid = model(data[t_], hid)
                if t_ >= data.shape[0] - readout_steps:
                    _, predictions = torch.max(output.data, 1)
                    # collect the correct predictions for each class
                    for lab, prediction in zip(label, predictions):
                        if lab == prediction:
                            correct_pred[classes[lab]] += 1
                        total_pred[classes[lab]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {} is: {:.1f} %".format(classname, accuracy))
