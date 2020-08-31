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

# Part 1: select data

speaker_N = 2

raw_data_path = 'data/2-speaker-encoded'
raw_list_path = 'data/2-speaker-firsr-samples-text_list_for_synthesis.txt'

def str2list(s):
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    r = []
    i = 1
    while i < len(s):
        r.append(str(s[i]))
        i += 3
    return r

class Encoding:
    def __init__(self, p, language_id, vec, t_sne_vec=None):
        self.p = p
        self.language_id = language_id
        self.vec = vec
        self.t_sne_vec = t_sne_vec


raw_data = []
data = []

f = open(raw_list_path, 'r', encoding='utf8').readlines()
f = [i.strip('\t\r\n') for i in f]

for i, s in enumerate(f):
    if i % speaker_N != 0:
        continue

    path = os.path.join(raw_data_path, 'encoded-batch_'+str(i)+'_sentence_0.npy')
    feature_seq = np.load(path)
    
    a = s.split('|')
    text = a[0] + '~'
    language_seq = str2list(a[1])
    
    assert len(text) == len(language_seq) and len(text) == feature_seq.shape[0]
    
    for i in range(len(text)):
        encoding = Encoding(text[i], language_seq[i], feature_seq[i])
        # not use raw_data, directly all is selected data
        data.append(encoding)


# Part 2: use t-SNE to reduct dimension

limit_sentences = random.randint(1, 1000000)
log_dir = '2-speaker-Encoded_Easy_t-SNE_RandomTrain_'+str(limit_sentences)+'_log_for2020-4-30-ppt'
os.makedirs(log_dir, exist_ok=True)

_pad        = '_'
_eos        = '~'
_characters = 'abcdefghijklmnopqrstuvwxyz123456!\',.;? []()'
# Export all symbols:
symbols = [_pad, _eos] + list(_characters)
syl_to_no = {}
for i, x in enumerate(symbols):
    syl_to_no[x] = i
print(len(symbols), symbols)

color_num = len(symbols) * 2
colors = cm.rainbow(np.linspace(0, 1, color_num))


def get_color(p, language_id):
    no = syl_to_no[p]
    if language_id == '0':
        no += len(symbols)
    c = colors[no]
    return c


def get_symbol(p, language_id):
    if language_id == '0':
        # change symbol to English version
        # A-Z, ' ' -> '*', '~' -> '=', 'biaodian'
        p = p.upper()
        if p == ' ':
            p = '*'
        if p == '~':
            p = '-'
    else:
        # a-z, '1-5', ' ', '~'
        if p == ' ':
            p = '^'
    return p


def plot_embedding_2d(encodings, title=None, save_path=None):
    X = [x.t_sne_vec for x in encodings]
    X = np.asarray(X)
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ch = get_symbol(encodings[i].p, encodings[i].language_id)
        c = get_color(encodings[i].p, encodings[i].language_id)
        ax.text(X[i, 0], X[i, 1], ch, color=c,
                fontdict={'weight': 'bold', 'size': 9})

    if title is not None:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=300)
    plt.close()


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


# Part 3: draw png

tsne_embedding_path = os.path.join(log_dir, 'full-tsne-encoded.png')
plot_embedding_2d(data, "t-SNE 2D", tsne_embedding_path)





