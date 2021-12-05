import torch
from utils.data_util import char_list
from utils.train_util import data_init, model_init


def eval_total_acc(config):
    # initialize data loaders
    test_loader = data_init(mode='test', use_velocity=config.use_velocity,
                            t_scale=config.t_scale, batch_s=config.batch_s,
                            context=config.context, context_w=config.context_w)
    # initialize model
    inp_size = 3
    if config.context is not None:
        inp_size += config.context_w
    model = model_init(mode='test', model_type=config.rnn_type,
                       input_size=inp_size, hidden_size=config.hidden_size,
                       save_path=config.save_path)

    correct = 0
    total = 0
    with torch.no_grad():
        for test_data, test_label in test_loader:
            hid = model.init_hidden(config.batch_s)
            for t_ in range(test_data.shape[0]):
                output, hid = model(test_data[t_], hid)
                if t_ >= test_data.shape[0] - config.readout_steps:
                    _, predicted = torch.max(output.detach(), 1)
                    total += test_label.size(0)
                    correct += (predicted == test_label).sum().item()
    # Accuracy and loss of the network on the test set:
    test_acc = 100 * correct / total
    print("Test Accuracy is: {:.1f} %".format(test_acc))
    return test_acc


def eval_class_acc(config):
    # initialize data loaders
    test_loader = data_init(mode='test', use_velocity=config.use_velocity,
                            t_scale=config.t_scale, batch_s=config.batch_s,
                            context=config.context, context_w=config.context_w)

    # initialize model
    inp_size = 3
    if config.context is not None:
        inp_size += config.context_w
    model = model_init(mode='test', model_type=config.rnn_type,
                       input_size=inp_size, hidden_size=config.hidden_size,
                       save_path=config.save_path)

    # prepare to count predictions for each class
    classes = char_list
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for test_data, test_label in test_loader:
            hid = model.init_hidden(config.batch_s)
            for t_ in range(test_data.shape[0]):
                output, hid = model(test_data[t_], hid)
                if t_ >= test_data.shape[0] - config.readout_steps:
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
