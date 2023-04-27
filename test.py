import torch
import torchvision
import argparse
import os
from model import Net
from torchvision import datasets
import numpy as np

parser = argparse.ArgumentParser(description="Test on veri")
parser.add_argument("--data-dir", default='data/VeRi', type=str)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = args.data_dir
query_dir = os.path.join(root, "query_test")
gallery_dir = os.path.join(root, "test")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 64)), torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
queryloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(query_dir, transform=transform),
                                          batch_size=64, shuffle=False, num_workers=4)
galleryloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
                                            batch_size=64, shuffle=False, num_workers=4)

net = Net(reid=True)
assert os.path.isfile("checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
print('Loading from checkpoint/ckpt.t7')
checkpoint = torch.load("checkpoint/ckpt.t7")
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict, strict=False)
net.eval()
net.to(device)

query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()

with torch.no_grad():
    for idx, (inputs, labels) in enumerate(queryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        query_features = torch.cat((query_features, features), dim=0)
        query_labels = torch.cat((query_labels, labels))
    for idx, (inputs, labels) in enumerate(galleryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        gallery_features = torch.cat((gallery_features, features), dim=0)
        gallery_labels = torch.cat((gallery_labels, labels))

gallery_labels -= 2 # for testing

### https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/evaluate.py
def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

image_datasets = {x: datasets.ImageFolder(os.path.join(root, x)) for x in ['test','query_test']}
gallery_path = image_datasets['test'].imgs
query_path = image_datasets['query_test'].imgs
gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf
    score = np.dot(gf, query)
    #score = np.dot(query, gf.t())
    index = np.argsort(score)
    index = index[::-1]
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc

if __name__ ==  '__main__':
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    ap = 0.0
    for i in range(len(query_labels)):
        ap_tmp, CMC_tmp = evaluate(query_features[i], query_labels[i], query_cam[i], gallery_features, gallery_labels, gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC/len(query_labels)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_labels)))
    features = {"qf": query_features, "ql": query_labels, "qc": query_cam, "gf": gallery_features, "gl": gallery_labels, "gc": gallery_cam}
    torch.save(features, "features.pth")