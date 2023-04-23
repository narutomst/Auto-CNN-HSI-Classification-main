import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
from numpy.core._multiarray_umath import ndarray

import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils

import torch.backends.cudnn as cudnn
import scipy.io as sio

from model import NetworkHSI
from sklearn.metrics import confusion_matrix
from data_prepare import read_data
import global_variable as glv

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=9, help='classes of HSI dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--infer_batch_size', type=int, default=100, help='infer batch size')
parser.add_argument('--learning_rate', type=float, default=0.004, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')  # 原版是'%m/%d %I:%M:%S %p'
fh = logging.FileHandler('./result/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# nband=103, nclass=9 .
# image_file = r'C:\Matlab练习\duogun\Pavia.mat'     
# label_file = r'C:\Matlab练习\duogun\Pavia_gt.mat'

# 本文件中的数据集地址不需要手动设置，由程序直接从global_variable.py模块中提取
# 以保证此处的地址与HSI_search.py中的一致。
glv._init()
image_file = glv.get_value('image_file')
label_file = glv.get_value('label_file')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(genotype, seed):
    data, shuffle_number = read_data(image_file, label_file, train_nsamples=200, validation_nsamples=100,
                                     windowsize=1, istraining=True, rand_seed=seed)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    model = NetworkHSI(args.init_channels, args.num_class, args.layers, genotype)   # model.py: 63
    model = model.to(device)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.5)

    min_val_obj = 100
    for epoch in range(args.epochs):
        tic = time.time()

        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # training      tar:target  pre:predict
        train_acc, train_obj, tar, pre = train(data.train, model, criterion, optimizer)
        # validation    tar_v:target_valid  pre_v:predictt_valid
        valid_acc, valid_obj, tar_v, pre_v = infer(data.validation, model, criterion)
        scheduler.step()
        toc = time.time()

        logging.info('Epoch %03d: train_loss = %f, train_acc = %f, val_loss = %f, val_acc = %f, time = %f', epoch + 1,
                     train_obj, train_acc, valid_obj, valid_acc, toc - tic)

        if epoch > args.epochs * 0.8 and valid_obj < min_val_obj:
            min_val_obj = valid_obj
            utils.save(model, './result/weights.pt')

    utils.load(model, './result/weights.pt')
    matrix = test_model(model, shuffle_number, seed)

    return matrix


def train(train_data, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    tar = np.array([])
    pre = np.array([])

    total_batch = int(train_data.num_examples / args.batch_size)
    for i in range(total_batch):
        input, target = train_data.next_batch(args.batch_size)
        input = torch.from_numpy(input).to(device)
        target = torch.from_numpy(np.argmax(target, axis=1)).to(device)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip) # 原版报错
        # 剪辑可迭代参数的梯度范数。
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))
        n = input.shape[0]
        # objs.update(loss.data[0], n)    # 原版报错 {IndexError}
        objs.update(loss.item(), n)
        # top1.update(prec1[0].data[0], n)    # 原版报错 {IndexError}
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


def infer(valid_data, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()    # 等价于 self.train(mode=False)
    tar = np.array([])
    pre = np.array([])

    total_batch = valid_data.num_examples // args.infer_batch_size
    for i in range(total_batch):
        input, target = valid_data.next_batch(args.infer_batch_size)

        input = torch.from_numpy(input).to(device)
        target = torch.from_numpy(np.argmax(target, axis=1)).to(device)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))
        n = input.shape[0]
        # objs.update(loss.data[0], n)    # 原版报错 {IndexError}
        objs.update(loss.item(), n)
        # top1.update(prec1[0].data[0], n)    # 原版报错 {IndexError}
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


def test_model(model, shuffle_number, seed):
    model.eval()    # 等价于 self.train(mode=False)
    total_tar = np.array([])
    total_pre = np.array([])

    i = 0
    test_nsamples = 0
    while 1:
        data = read_data(image_file, label_file, train_nsamples=200, validation_nsamples=100,
                         windowsize=1, istraining=False, shuffle_number=shuffle_number, times=i, rand_seed=seed)
        if data is None:
            matrix = confusion_matrix(total_tar, total_pre)
            return matrix

        test_nsamples += data.test.num_examples

        x_test, y_test = data.test.images, data.test.labels
        add_samples = args.batch_size - data.test.num_examples % args.batch_size
        x_test = np.concatenate((x_test[0:add_samples, :, :, :], x_test), axis=0)
        y_test = np.concatenate((y_test[0:add_samples, :], y_test), axis=0)

        tar = np.array([])
        pre = np.array([])
        total_batch = (data.test.num_examples + add_samples) // args.batch_size
        for j in range(total_batch):
            input, target = x_test[j * args.batch_size:(j + 1) * args.batch_size, :, :, :], \
                            y_test[j * args.batch_size:(j + 1) * args.batch_size, :]

            input = torch.from_numpy(input).to(device)
            target = torch.from_numpy(np.argmax(target, axis=1)).to(device)

            logits = model(input)

            _, t, p = utils.accuracy(logits, target, topk=(1,))
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())

        total_tar = np.append(total_tar, tar[add_samples:])
        total_pre = np.append(total_pre, pre[add_samples:])
        i = i + 1


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    # AA = np.zeros([shape[0]], dtype=float)
    TPR: ndarray = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        TPR[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA: float = number / np.sum(matrix)
    AA: float = np.mean(TPR)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA, Kappa, TPR


if __name__ == '__main__':
    genotype = eval('genotypes.{}'.format(args.arch))
    # 通过在字符串对象上调用format()方法来进行字符串格式化，使字符串格式化的语法更加规范。
    # args.arch: 'HSI'
    # format(args.arch): 'HSI'
    # 'genotypes.{}'.format(args.arch): 'genotypes.HSI'
    # 整个语句的真正表达式为：
    # genotype = genotypes.HSI
    matrix = main(genotype=genotype, seed=np.random.randint(low=0, high=10000, size=1))
    # mian()位于 line 49
    # OA, AA_mean, Kappa, AA = cal_results(matrix)
    OA, AA, Kappa, TPR = cal_results(matrix)
    print(OA)

    # 将分类结果按列保存到txt中
    now = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
    resultStr = '#  OA,     AA,   Kappa,  TPR  ' + now + image_file

    # 不同数据集的分类结果保存到不同的文件中，fname
    dir_name, base_name = osp.split(image_file)
    f_name, f_ext = osp.splitext(base_name)
    fname = './result/classification_' + f_name + '.txt'

    a = list([OA, AA, Kappa])
    a.extend(list(TPR))
    # X = np.array(a)
    # 使用f和np.savetxt()组合来保存数据，格式处理很方便
    # 数据的标题只写一次，只要不是空文件，追加的时候就不用再写标题了
    # 数据文件fname不存在的话，open()就会自动创建fname文件
    with open(fname, mode='a', encoding='utf-8') as f:  # 'utf-8'可确保中文不乱码;如果fname不存在，则会创建fname文件
        # if os.path.getsize(fname) == 0:
        #     header = resultStr.rstrip('\n')
        # else:
        #     header = ''
        np.savetxt(f, np.column_stack(a), fmt='%.4f', delimiter=' ', newline='\n', header=resultStr,
                   footer='', comments='', encoding=None)
        f.close()

    # 单纯使用f.write()保存数据，格式处理上比较麻烦
    # b = ['%.4f ' % x for x in a]
    # with open(fname, mode='a', encoding='utf-8') as f:
    #     f.write('\n')
    #     for i in range(0, len(b)):
    #         f.write(b[i])
    #     f.close()


