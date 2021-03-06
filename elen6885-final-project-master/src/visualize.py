# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})


def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(datas) < bin_size:
        print('len of data: %d, len of result: %d'%(len(datas), len(result)))
    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)

    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]


def visdom_plot(viz, win, folder, game, name, bin_size=100, smooth=1, losses=None):
    tx, ty = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        print('returning early')
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))

    # Ugly hack to detect atari
    if game.find('NoFrameskip') > -1:
        plt.xticks([1e6, 2e6, 4e6, 6e6, 8e6, 10e6],
                   ["1M", "2M", "4M", "6M", "8M", "10M"])
        plt.xlim(0, 10e6)
    else:
        #plt.xticks([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 10e5],
        #           ["0.1M", "0.2M", "0.3M", "0.4M", "0.5M", "0.6M", "0.7M", "0.8M", "0.9M", "1.0M"])
        #plt.xlim(0, 1e6)
        plt.xticks([1e5, 5e5, 10e5, 15e5, 20e5, 25e5, 30e5, 35e5, 40e5, 45e5, 50e5],
                   ["0.1M", "0.5M", "1M", "1.5M", "2M", "2.5M", "3M", "3.5M", "4M", "4.5M", "5M"])
        plt.xlim(0, 5e6)

    plt.xlabel('Number of Timesteps')
    if losses == None:
        plt.ylabel('Rewards')
    else:
        plt.ylabel('Teacher Network Rewards')
    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    if losses == None:
        return viz.image(image, win)
    else:
        win[0] = viz.image(image, win[0])

    losses = np.array(losses)
    x = np.arange(losses.size)
    fig = plt.figure()
    plt.plot(x, losses, label="loss")
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Loss')
    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)
    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win[1] = viz.image(image, win[1])
    print('plotted')
    return win

def visdom_data_plot(viz, win, game, name, data, ylabel, bin_size=100, smooth = 1):

    data = np.array(data)[:, 0]
    x = np.arange(data.shape[0])
    
    if data.shape[0] < bin_size:
        return win

    if smooth == 1:
        x, data = smooth_reward_curve(x, data)
    x, data = fix_point(x, data, bin_size)
    
    fig = plt.figure()
    plt.plot(x, data, label=ylabel)
    plt.xlabel('Number of Timesteps')
    plt.ylabel(ylabel)
    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()
    
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)
    
    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win = viz.image(image, win)
    print('plotted')
    return win

if __name__ == "__main__":
    from visdom import Visdom
    viz = Visdom()
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
