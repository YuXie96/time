import logging
import os.path as osp

import torch
import torch.nn as nn

from configs.config_global import USE_CUDA, LOG_LEVEL
from utils.config_utils import load_config
from utils.logger import Logger
from utils.train_util import data_init, model_init, grad_clipping


def train_from_path(path):
    """Train from a path with a config file in it."""
    logging.basicConfig(level=LOG_LEVEL)
    config = load_config(path)
    model_train(config)


def model_train(config):
    if USE_CUDA:
        logging.info("training with GPU")

    # initialize logger
    logger = Logger(output_dir=config.save_path,
                    exp_name=config.experiment_name)

    # gradient clipping
    if config.grad_clip is not None:
        logging.info("Performs grad clipping with max norm {}" + str(config.grad_clip))

    # initialize data loaders
    train_loader = data_init(mode='train', use_velocity=config.use_velocity, batch_s=config.batch_s)
    test_loader = data_init(mode='test', use_velocity=config.use_velocity, batch_s=config.batch_s)

    # initialize model
    model = model_init(mode='train')

    # initialize training
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    i_b = 0
    i_log = 0
    testloss_list = []
    break_flag = False
    train_loss = 0.0
    for epoch in range(config.num_ep):
        for data, label in train_loader:
            # save model
            if i_b % config.save_every == 0:
                torch.save(model.state_dict(), osp.join(config.save_path, 'net_{}.pth'.format(i_b)))

            loss = 0.0
            hid = model.init_hidden(config.batch_s)
            for t_ in range(data.shape[0]):
                output, hid = model(data[t_], hid)
                if t_ >= data.shape[0] - config.readout_steps:
                    loss += criterion(output, label)
            optimizer.zero_grad()
            loss /= config.readout_steps
            loss.backward()
            # gradient clipping
            if config.grad_clip is not None:
                grad_clipping(model, 1.0)
            optimizer.step()
            train_loss += loss.item()

            # test and log performance
            if i_b % config.log_every == config.log_every - 1:
                correct = 0
                total = 0
                test_loss = 0.0
                test_b = 0
                with torch.no_grad():
                    for test_data, test_label in test_loader:
                        hid = model.init_hidden(config.batch_s)
                        for t_ in range(test_data.shape[0]):
                            output, hid = model(test_data[t_], hid)
                            if t_ >= test_data.shape[0] - config.readout_steps:
                                test_loss += criterion(output, test_label).item()
                                _, predicted = torch.max(output.detach(), 1)
                                total += test_label.size(0)
                                correct += (predicted == test_label).sum().item()
                        test_b += 1

                # Accuracy and loss of the network on the test set:
                avg_testloss = test_loss / (test_b * config.readout_steps)
                test_acc = 100 * correct / total

                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('BatchNum', i_b)
                logger.log_tabular('DataNum', i_b * config.batch_s)
                logger.log_tabular('TrainLoss', train_loss / config.log_every)
                logger.log_tabular('TestLoss', avg_testloss)
                logger.log_tabular('TestAcc', test_acc)
                logger.dump_tabular()

                train_loss = 0.0

                i_log += 1
                # check plateau
                testloss_list.append(avg_testloss)
                # save the model with best testing loss
                if avg_testloss <= min(testloss_list):
                    torch.save(model.state_dict(),
                               osp.join(config.save_path, 'net_best.pth'))

                if config.early_stop \
                        and (i_log >= 2 * config.eslen) \
                        and (min(testloss_list[-config.eslen:]) > min(testloss_list[:-config.eslen])):
                    break_flag = True
                    break

            i_b += 1
            if i_b >= config.max_batch:
                break_flag = True
                break

        if break_flag:
            break
