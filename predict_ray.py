import argparse
import os
import json
import torch
import ray
import ray.train as train
from ray.train.trainer import Trainer
from ray.train.callbacks import JsonLoggerCallback
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms, datasets
from tqdm import tqdm
from model import resnet34
import time
from typing import Dict
import time
import numpy as np
import matplotlib.pyplot as plt

def data_load(args):
    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = args.train_path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    car_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in car_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_index.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size  # batch_size
    nw = 8  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                                        transform=data_transform["val"])
    val_num=len(validate_dataset)
    if args.with_ray:
        train_loader = DataLoader(train_dataset,batch_size=batch_size,num_workers=nw)
        validate_loader = DataLoader(validate_dataset,batch_size=batch_size,num_workers=nw)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=nw)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size, shuffle=False,num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    return train_loader,validate_loader


def train_epoch(net,train_loader,optimizer,loss_function,epoch,epochs):
    size = len(train_loader.dataset)
    tr_loss = 0.0
    true = 0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data
        logits = net(images)
        loss = loss_function(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print statistics
        tr_loss += loss.item()*len(labels)
        true += (logits.argmax(1) == labels).type(torch.float).sum().item()
        acc=100*(logits.argmax(1) == labels).type(torch.float).sum().item()/len(labels)
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}%".format(epoch + 1, epochs,loss,acc)
    tr_acc = (true / size) * 100.
    tr_loss=tr_loss/ size
    return tr_loss, tr_acc, net

def valid_epoch(net,validate_loader,device,loss_function):
    # validate
    size = len(validate_loader.dataset)
    num_batches = len(validate_loader)
    net.eval()
    time1=time.time()
    acc = 0.0  # accumulate accurate number / epoch
    va_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            loss = loss_function(outputs, val_labels.to(device))
            va_loss+=loss.item()*len(val_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_acc = 100 * (outputs.argmax(1) == val_labels.to(device)).type(torch.float).sum().item() / len(val_labels)
            val_bar.desc = "loss:{:.3f} acc:{:.3f}%".format(loss,val_acc)
    time2=time.time()
    val_accurate = (acc /size)* 100.
    va_loss = va_loss/ size
#     if val_accurate > best_acc:
#         best_acc = val_accurate
#         torch.save(net.state_dict(), save_path)
    return va_loss,val_accurate,time2-time1


def train_validate_network_with_ray(config: Dict):
    args = config['args']
    tr_loader,va_loader = data_load(args)
#     tr_loader = train.torch.prepare_data_loader(tr_loader)
    va_loader = train.torch.prepare_data_loader(va_loader)
    
    net = resnet34()
    # load pretrain weights

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 1777)
    
    model_weight_path ="./resnet34_ray_new.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    checkpoint=torch.load(model_weight_path)
    net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
    device = torch.device(f"cuda:{train.local_rank()}" if args.use_gpu else "cpu")
    model = train.torch.prepare_model(net)
    
   

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path = args.save_path  
    tr_loss, tr_acc, va_loss, va_acc = [], [], [], []

    va_loss1, va_acc1,va_time = valid_epoch(model,va_loader,device,loss_function)


    return va_acc1,va_time



def plot_acc_loss(args, tr_loss, tr_acc, va_loss, va_acc):
    if not args.with_ray:
        plt.plot(tr_acc)
        plt.plot(va_acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'validation'])
        plt.savefig('./acc.png')
        plt.close()

        plt.plot(tr_loss)
        plt.plot(va_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['train', 'validation'])
        plt.savefig('./loss.png')
        plt.close()

    if args.with_ray:
        plt.plot(tr_acc)
        plt.plot(va_acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'validation1', 'train2', 'validation2'])
        plt.savefig('./acc_with_ray.png')
        plt.close()

        plt.plot(tr_loss)
        plt.plot(va_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['train1', 'validation1', 'train2', 'validation2'])
        plt.savefig('./loss_with_ray.png')
        plt.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resnet34 Transfer Learning Image Classification',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=180, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='epochs')
    parser.add_argument('--train-path', type=str, default='../../myData', help='dataset path')
    parser.add_argument('--save-path', type=str, default='./resnet34_ray_new.pth', help='model save path')
    parser.add_argument('--with-ray', type=bool, default=True, help='with Ray')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--use-gpu', type=bool, default=True, help='choose divice for training')
    args = parser.parse_args()

    start = time.time()
    if not args.with_ray:
        tr_loss, tr_acc, va_loss, va_acc = train_validate_network(args)
        end = time.time()
        print(f"Training and validation time: {end-start:>.2f}")
        plot_acc_loss(args, tr_loss, tr_acc, va_loss, va_acc)
    elif args.with_ray:
        if args.use_gpu:
            ray.init(num_gpus=args.num_workers)
        else:
            ray.init(num_cpus=args.num_workers)
        trainer = Trainer(backend="torch", num_workers=args.num_workers, use_gpu=args.use_gpu)
        trainer.start()
        res = trainer.run(train_func=train_validate_network_with_ray, config={'args':args}, callbacks=[JsonLoggerCallback()])
        trainer.shutdown()
        end = time.time()
        print(f"validation time: {end - start:.>2f}")
#         tr_loss, tr_acc, va_loss, va_acc = [], [], [], []
        va_acc=[]
        
        va_time=[]
        
        for idx_w in range(args.num_workers):
            tmp_res = res[idx_w]
            va_acc.append(tmp_res[0])
            va_time.append(tmp_res[1])
    
            
            
        print(f"Tvalidation1 time: {va_time[0]:>.2f}")
        print(f"Tvalidation1 acc: {va_acc[0]:>.2f}")
        print(f"Tvalidation2 time: {va_time[1]:>.2f}")
        print(f"Tvalidation2 acc: {va_acc[1]:>.2f}")
#             tr_loss.append(tmp_res[0])
#             tr_acc.append(tmp_res[1])
#             va_loss.append(tmp_res[2])
#             va_acc.append(tmp_res[3])
#         tr_loss = np.array(tr_loss)
#         tr_acc = np.array(tr_acc)
#         va_loss = np.array(va_loss)
#         va_acc = np.array(va_acc)
#         np.save('tr_loss_new.npy',tr_loss)
#         np.save('tr_acc_new.npy',tr_acc)
#         np.save('va_loss_new.npy',va_loss)
#         np.save('va_acc_new.npy',va_acc)
#         plot_acc_loss(args, tr_loss, tr_acc, va_loss, va_acc)




