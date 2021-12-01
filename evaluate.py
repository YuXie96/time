import torch
from data_util import char_list
from train_util import data_init, model_init


def eval_class_acc(config):
    # initialize data loaders
    test_loader = data_init(mode='test', use_velocity=config.use_velocity, batch_s=config.batch_s)

    # initialize model
    model = model_init(mode='test', save_path=config.save_path)

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
