import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils
from data_util import num2chr


def show_tra_3d(tra):
    ax = plt.axes(projection='3d')
    ax.plot3D(tra[0, :], tra[1, :], tra[2, :], 'gray')
    ax.set_xlabel('d1')
    ax.set_ylabel('d2')
    ax.set_zlabel('d3')


def show_batch(batch, tra_plot_func):
    tras, labels = batch
    assert len(labels) == tras.shape[1]
    batch_size = len(labels)

    for i_b in range(batch_size):
        print(i_b, 'shape:', tras[:, i_b, :].shape,
              'label:', labels[i_b], num2chr(labels[i_b]))
        plt.subplot(batch_size, 1, i_b + 1)
        tra_plot_func(tras[:, i_b, :].transpose(0, 1))
    plt.suptitle('Batch from dataloader')


def show_tra_scatter(tra):
    plt.plot(tra[0, :], tra[1, :], color='gray')
    color_range = list(range(tra.shape[1]))
    plt.scatter(tra[0, :], tra[1, :], c=color_range, cmap='Blues')
    plt.gca().set_aspect('equal')
    plt.colorbar()


def show_tra_batch_scatter(batch):
    show_batch(batch, show_tra_scatter)


def show_tra_heat(tra):
    plt.imshow(tra, aspect='auto', interpolation='none')
    plt.colorbar()


def show_tra_batch_heat(batch):
    show_batch(batch, show_tra_heat)



