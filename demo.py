import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
from torchvision import datasets
import os
import argparse
matplotlib.use('Agg')

### https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/demo.py
parser = argparse.ArgumentParser(description="Demo on veri")
parser.add_argument("--data-dir", default='data/VeRi', type=str)
parser.add_argument("--index-test", default=0, type=int)
args = parser.parse_args()

root = args.data_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(root, x)) for x in ['test','query_test']}
features = torch.load("features.pth")
query_features = features["qf"]
query_labels = features["ql"]
query_cam = features["qc"]
gallery_features = features["gf"]
gallery_labels = features["gl"]
gallery_cam = features["gc"]

def imshow(path, title=None):
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.show()

def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

i = args.index_test
index = sort_img(query_features[i], query_labels[i], query_cam[i], gallery_features, gallery_labels, gallery_cam)

query_path, _ = image_datasets['query_test'].imgs[i]
query_label = query_labels[i]
print(query_path)
print('Top 10 images are as follow:')
try:
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query_test')
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets['test'].imgs[index[i]]
        label = gallery_labels[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
fig.savefig("demo.jpg")