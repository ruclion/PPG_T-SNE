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
from sklearn.decomposition import PCA
import random
from sklearn.externals import joblib
import time
import pickle as pkl
from datetime import datetime

np.random.seed(0)

NUM_SELECTED = 8
speaker_N = 1
language_N = 1
raw_data_path1 = '/home/hujk17/PPG2MEL_DATA/LJSpeech-1.1_Norm_Sort/norm_ppg'
raw_list_path1 = '/home/hujk17/PPG2MEL_DATA/LJSpeech-1.1_Norm_Sort/sorted_train.txt'
raw_data_path2 = '/home/hujk17/ppgs_extractor/DataBaker_Bilingual_CN/ppg_from_generate_batch'
raw_list_path2 = '/home/hujk17/ppgs_extractor/DataBaker_Bilingual_CN/meta.txt'

# limit_sentences = random.randint(1, 1000000)
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
log_dir = 'T-SNE_'+STARTED_DATESTRING+'_log_for_2020-8-31'
# debug_limit_sentences = 202128
# log_dir = 'T-SNE_'+str(debug_limit_sentences)+'_log_for_2020-8-31'
os.makedirs(log_dir, exist_ok=True)

class Point:
    def __init__(self, t_sne_vec, speaker_id, language_id, vec, path, idx):
        self.t_sne_vec = t_sne_vec
        self.speaker_id = speaker_id
        self.language_id = language_id
        self.vec = vec
        self.path = path
        self.idx = idx


def randome_select(raw_list_path, raw_data_path, speaker, language, NUM_SELECTED, data):
    if raw_list_path == raw_list_path1:
        f = open(raw_list_path, 'r', encoding='utf8').readlines()
        f = [i.strip() for i in f]
    elif raw_list_path == raw_list_path2:
        f = open(raw_list_path, 'r', encoding='utf8').readlines()
        len_f = len(f)
        a = []
        for i in range(len_f):
            if i % 2 == 0:
                x = f[i].strip()[0:6]
                a.append(x)
        f = a
        
    np.random.shuffle(f)
    f = f[:NUM_SELECTED]
    print(f)

    # 准备
    for i, s in enumerate(f):
        path = os.path.join(raw_data_path, s + '.npy')
        ppg = np.load(path)
        
        t_sne_vec = None
        speaker_id = speaker
        language_id = language
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
        # https://www.cnblogs.com/zinyy/p/9333349.html
        print('start PCA')
        pca = PCA(n_components=30)
        X = pca.fit_transform(X)
        print('end PCA')
        print("Computing t-SNE encoded")
        print('starting...')
        start = time.time()
        tsne = manifold.TSNE(n_components=2, perplexity=30.0, init='pca', random_state=0, verbose=1)
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

    # 计算颜色和字母
    class_tag = []
    for x in points:
        if x.speaker_id == 0 and x.language_id == 0:
            class_tag.append(('r', 'x'))
        elif x.speaker_id == 1 and x.language_id == 1:
            class_tag.append(('g', 'o'))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], class_tag[i][1], color=class_tag[i][0],
                fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)
    plt.close()


def select_calcu_draw():
    data = []
    data = randome_select(raw_list_path1, raw_data_path1, 0, 0, NUM_SELECTED, data)
    data = randome_select(raw_list_path2, raw_data_path2, 1, 1, NUM_SELECTED, data)
    data = calcu_tsne(data)
    plot_embedding_2d(data, "t-SNE 2D", os.path.join(log_dir, 'ppg-tsne.png'))
        

if __name__ == "__main__":
    select_calcu_draw()

