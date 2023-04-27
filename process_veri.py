import os
from shutil import copyfile

train_path = 'data/VeRi/train_resize'
train_save_path = 'data/VeRi/train'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/v' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

test_path = 'data/VeRi/test_resize'
test_save_path = 'data/VeRi/test'
if not os.path.isdir(test_save_path):
    os.mkdir(test_save_path)
    for root, dirs, files in os.walk(test_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = test_path + '/' + name
            dst_path = test_save_path + '/v' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

query_path = 'data/VeRi/query_resize'
query_save_path = 'data/VeRi/query_test'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/v' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)