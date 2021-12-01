import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms

from data_util import CharacterTrajectoriesDataset, pad_collate,\
    SkipTransform, TrimZeros, CumSum, char_list
from train_util import grad_clipping
from models import RNNModel


if __name__ == '__main__':
    readout_steps = 1
    num_classes = 20
    batch_s = 10
    use_velocity = True

    if use_velocity:
        # classification based on velocity
        trans = transforms.Compose([TrimZeros(),
                                    SkipTransform(skip_num=2)])
    else:
        # classification based on trajectory
        # TODO: should normalize trajectory
        trans = transforms.Compose([TrimZeros(),
                                    SkipTransform(skip_num=2),
                                    CumSum()])

    train_data = CharacterTrajectoriesDataset(large_split=True, transform=trans)
    test_data = CharacterTrajectoriesDataset(large_split=False, transform=trans)

    train_loader = DataLoader(train_data, batch_size=batch_s, shuffle=True,
                              collate_fn=pad_collate, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_s, shuffle=False,
                             collate_fn=pad_collate, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    model = RNNModel(3, 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters())

    for ep in range(20):
        for data, label in train_loader:
            loss = 0.0
            hid = model.init_hidden(batch_s)
            for t_ in range(data.shape[0]):
                output, hid = model(data[t_], hid)
                if t_ >= data.shape[0] - readout_steps:
                    loss += criterion(output, label)
            optimizer.zero_grad()
            loss /= readout_steps
            loss.backward()
            grad_clipping(model, 1.0)
            optimizer.step()

        # TODO: here is only printing loss from the latest trial
        print(loss.item())
        # testing at the end of each epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for test_data, test_label in test_loader:
                hid = model.init_hidden(batch_s)
                for t_ in range(test_data.shape[0]):
                    output, hid = model(test_data[t_], hid)
                    if t_ >= test_data.shape[0] - readout_steps:
                        _, predicted = torch.max(output.detach(), 1)
                        total += test_label.size(0)
                        correct += (predicted == test_label).sum().item()
        print('Accuracy of the network on the test set: %d %%' % (
                100 * correct / total))

    # prepare to count predictions for each class
    # classes = list(range(num_classes))
    classes = char_list
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for test_data, test_label in test_loader:
            hid = model.init_hidden(batch_s)
            for t_ in range(test_data.shape[0]):
                output, hid = model(test_data[t_], hid)
                if t_ >= test_data.shape[0] - readout_steps:
                    _, predictions = torch.max(output.detach(), 1)
                    # collect the correct predictions for each class
                    for lab, prediction in zip(test_label, predictions):
                        if lab == prediction:
                            correct_pred[classes[lab]] += 1
                        total_pred[classes[lab]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {} is: {:.1f} %".format(classname, accuracy))
