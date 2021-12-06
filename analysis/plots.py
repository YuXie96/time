import os.path as osp

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from configs.config_global import FIG_DIR

plt.rcParams.update({'font.size': 16})
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

line_styles = ['-', '--', ':']
# Default colors
# colors = ['red', 'tomato', 'green', 'lightgreen', 'blue', 'lightblue']


# Plot testing loss recorded during training
def plot_train_log_loss(data, exp_name):
    plt.figure()
    legends = []
    for i_, datum in enumerate(data):
        model_n, data_f = datum
        x_axis = data_f['DataNum']
        trainloss = data_f['TrainLoss']
        testloss = data_f['TestLoss']
        plt.plot(x_axis, trainloss, color='C'+str(i_), linestyle='--')
        plt.plot(x_axis, testloss, color='C'+str(i_))
        legends += [model_n + " Train Loss"]
        legends += [model_n + " Test Loss"]

    plt.legend(legends, loc='best', fontsize=6)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.xlim([0, 4e4])
    plt.ylim([0, 3])
    plt.xlabel('Train Data')
    plt.ylabel('Loss')

    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(osp.join(FIG_DIR, exp_name + 'train_log_loss.pdf'), transparent=True)


# Plot testing accuracy recorded during training
def plot_train_log_acc(data, exp_name):
    plt.figure()
    legends = []
    for i_, datum in enumerate(data):
        model_n, data_f = datum
        x_axis = data_f['DataNum']
        testacc = data_f['TestAcc']
        plt.plot(x_axis, testacc, color='C'+str(i_))
        legends += [model_n + " Test Acc"]

    plt.legend(legends, loc='best', fontsize=6)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.xlim([0, 4e4])
    plt.ylim([0, 100])
    plt.xlabel('Train Data')
    plt.ylabel('Accuracy')

    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(osp.join(FIG_DIR, exp_name + 'train_log_acc.pdf'), transparent=True)


def plot_gen(tscales, accs, append_str):
    plt.figure()
    plt.plot(tscales, accs)

    plt.ylim([0, 100])
    plt.xlabel('Time Scales')
    plt.ylabel('Test Accuracy')

    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(osp.join(FIG_DIR, 'time_scale_gen' + append_str + '.pdf'), transparent=True)


def plot_group_gen(configs):
    legend_list = []
    t_scale_list = []
    acc_list = []

    for cfg in configs:
        t_scale_list.append(np.load(osp.join(cfg.save_path, 'tscalelist.npy')))
        acc_list.append(np.load(osp.join(cfg.save_path, 'acclist.npy')))
        legend_list.append(cfg.rnn_type)

    plt.figure()
    for t_scale, acc in zip(t_scale_list, acc_list):
        plt.plot(t_scale, acc)

    plt.ylim([0, 100])
    plt.xlabel('Time Scales')
    plt.ylabel('Test Accuracy')
    plt.legend(legend_list)

    ax = plt.gca()
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(osp.join(FIG_DIR, 'time_scale_gen_all.pdf'), transparent=True)
