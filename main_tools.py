import os
import sys
import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import (manifold, datasets, decomposition,
                     ensemble, random_projection)
import random
from sklearn.externals import joblib
import time
import pickle as pkl

np.random.seed(0)

NUM_SELECTED = 2
speaker_N = 1
language_N = 1
raw_data_path = '/home/hujk17/PPG2MEL_DATA/LJSpeech-1.1_Norm_Sort/norm_ppg'
raw_list_path = '/home/hujk17/PPG2MEL_DATA/LJSpeech-1.1_Norm_Sort/sorted_train.txt'

limit_sentences = random.randint(1, 1000000)
log_dir = 'T-SNE_'+str(limit_sentences)+'_log_for_2020-8-31'
os.makedirs(log_dir, exist_ok=True)

class Point:
    def __init__(self, t_sne_vec, speaker_id, language_id, vec, path, idx):
        self.t_sne_vec = t_sne_vec
        self.speaker_id = speaker_id
        self.language_id = language_id
        self.vec = vec
        self.path = path
        self.idx = idx


def randome_select(raw_list_path, NUM_SELECTED, data):
    f = open(raw_list_path, 'r', encoding='utf8').readlines()
    f = [i.strip() for i in f]
    np.random.shuffle(f)
    f = f[:NUM_SELECTED]

    # 准备
    for i, s in enumerate(f):
        path = os.path.join(raw_data_path, s + '.npy')
        ppg = np.load(path)
        
        t_sne_vec = None
        speaker_id = 0
        language_id = 0
        path = path
        for i in range(ppg.shape[0]):
            vec = ppg[i]
            idx = i
            p = Point(t_sne_vec, speaker_id, language_id, vec, path, idx)
            data.append(p)
    print('tot ppg:', len(data))
    return data


def calcu_tsne(data):
    # 计算
    X_tsne_path = os.path.join(log_dir, 'X_tsne.npy')
    if os.path.exists(X_tsne_path):
        print("Use Computed t-SNE encoded")
        X_tsne = np.load(X_tsne_path)
    else:
        X = [a.vec for a in data]
        X = np.asarray(X)
        print("Computing t-SNE encoded")
        print('starting...')
        start = time.time()
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, verbose=1)
        X_tsne = tsne.fit_transform(X)
        print('ending... time consuming is:', time.time() - start)
        np.save(X_tsne_path, X_tsne)

    for i, a in enumerate(X_tsne):
        data[i].t_sne_vec = X_tsne[i]
    return data

def plot_embedding_2d(points, title=None, save_path=None):
    # 计算坐标
    X = [x.t_sne_vec for x in points]
    X = np.asarray(X)
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    # 计算颜色
    class_color = []
    for x in points:
        if x.speaker_id == 0 and x.language_id == 0:
            class_color.append('r')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], color=class_color[i])

    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)
    plt.close()


def select_calcu_draw():
    data = []
    data = randome_select(raw_list_path, NUM_SELECTED, data)
    data = calcu_tsne(data)
    plot_embedding_2d(data, "t-SNE 2D", os.path.join(log_dir, 'ppg-tsne.png'))
        

if __name__ == "__main__":
    select_calcu_draw()

