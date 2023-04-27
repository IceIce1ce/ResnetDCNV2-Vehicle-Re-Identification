import argparse
import os
import time
import torch
import torchvision
from model import Net
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="Train on veri")
parser.add_argument("--data-dir", default='data/VeRi', type=str)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--interval",'-i', default=20, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")
transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop((128, 64), padding=4), torchvision.transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_test = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 64)), torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
trainloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(train_dir, transform=transform_train), batch_size=128, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(test_dir, transform=transform_test), batch_size=128, shuffle=False, num_workers=4)
num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))

total_epoch = 40
net = Net(num_classes=num_classes)
net.to(device)
best_acc = 0.0
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    print("\nEpoch : %d"%(epoch+1))
    net.train()
    training_loss = 0.0
    train_loss = 0.0
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(labels.data).cpu().sum()
        total += labels.size(0)
        if (idx+1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                  100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, (100.0*correct)/total))
            training_loss = 0.0
            start = time.time()
    return train_loss/len(trainloader), 1.0 - correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(labels.data).cpu().sum()
            total += labels.size(0)
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
              100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, (100.0*correct)/total))
    acc = (100.0*correct)/total
    if acc > best_acc:
        best_acc = acc
        print('Best acc:', best_acc.item())
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {'net_dict':net.state_dict(), 'acc':best_acc, 'epoch':epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, 'checkpoint/ckpt.t7')
    return test_loss/len(testloader), 1.0 - correct/total

def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():
    for epoch in range(total_epoch):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        print('Train loss:', train_loss)
        print('Train error:', train_err.item())
        print('Test loss:', test_loss)
        print('Test error:', test_err.item())
        if (epoch + 1) % 20 == 0:
            lr_decay()

if __name__ == '__main__':
    start = time.time()
    main()
    print('Best acc:', best_acc.item())
    print('Total training time:', time.time() - start)