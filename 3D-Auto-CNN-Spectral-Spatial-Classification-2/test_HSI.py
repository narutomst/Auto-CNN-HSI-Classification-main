import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch.utils.data
import torch
from numpy.core._multiarray_umath import ndarray
import utils
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio

from model import NetworkHSI
from sklearn.metrics import confusion_matrix
from utils import cutout
from data_prepare import read_data, load_data
import global_variable as glv

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=9, help='classes of HSI dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.016, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=2, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--num_cut', type=int, default=10, help='band cutout')
parser.add_argument('--Train', type=int, default=200, help='Train_num')
parser.add_argument('--Valid', type=int, default=100, help='Valid_num')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.manualSeed = random.randint(1, 10000)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%Y-%m-%d %H:%M:%S')  # '%m/%d %I:%M:%S %p'
fh = logging.FileHandler('./result/log_3D.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# read data
# image_file = r'C:\Matlab练习\duogun\PaviaU.mat'
# label_file = r'C:\Matlab练习\duogun\PaviaU_gt.mat'

# 本文件中的数据集地址不需要手动设置，由程序直接从global_variable.py模块中提取
# 以保证此处的地址与HSI_search.py中的一致。
glv._init()
image_file = glv.get_value('image_file')
label_file = glv.get_value('label_file')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(genotype, seed, cut=False):

    windowsize = 32
    batchnumber = 1000
    image, label = load_data(image_file, label_file)
    # 取得HSI数据尺寸
    [nRow, nColumn, nBand] = image.shape
    # 取得地物类别数量
    num_class = int(np.max(label))
    # windowsize = 32
    HalfWidth = windowsize // 2

    mask = np.zeros([nRow, nColumn])
    mask[HalfWidth: -1 - HalfWidth + 1, HalfWidth: -1 - HalfWidth + 1] = 1
    # mask[17:-16, 17:-16] = 1, 负索引 i 的含义是从数组的末尾开始计数(
    # 即，如果i < 0 ，被解释为 n + i，其中 n 是相应维度中的元素数量
    # row=610, col=340, 则上面的切片表达式被解释为mask[17:610-16, 17:340-16] = 1
    # 并且，numpy中的切片索引是计头不计尾，即i:j 表示i,i+1,...,(j-1)
    # 也就是说，整幅图片的上下左右四个方向上，边缘的16行、16列被裁剪掉了。
    label = label * mask  # 对应元素相乘 element-wise product: np.multiply(), 或 *
    # 返回G中非零元素的行索引和列索引值
    [non_zero_row, non_zero_col] = label.nonzero()
    # 统计整张HSI图片上的非零label的样本总数。
    # 将以下关键变量的名称与data_prepare中保持一致
    number_samples = np.size(non_zero_row)
    np.random.seed(seed)
    shuffle_number = np.random.permutation(number_samples)
    train_nsamples = args.Train
    validation_nsamples = args.Valid
    test_nsamples = (number_samples - train_nsamples - validation_nsamples)

    batchtr = train_nsamples
    numbatch1 = train_nsamples // batchtr
    # batchnumber = 1000
    numbatch2 = test_nsamples // batchnumber

    # 散装语句到此结束

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    args.cutout = cut  # False
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.manualSeed)

    model = NetworkHSI(nBand, args.init_channels, num_class, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 5, 0.5)

    min_val_obj = 100
    for epoch in range(args.epochs):
        tic = time.time()
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        # 初始化dict变量imdb{}
        imdb = {'data': np.zeros([windowsize, windowsize, nBand, train_nsamples + validation_nsamples],
                                 dtype=np.float32),
                'Labels': np.zeros([train_nsamples + validation_nsamples], dtype=np.int64),
                'set': np.hstack((np.ones([train_nsamples]), 3 * np.ones([validation_nsamples]))).astype(np.int64)}
        # imdb['data'].shape: (32, 32, 103, 300); imdb['Labels'].shape:(300,),表示一维数组; imdb['set'].shape:(300,),表示一维数组
        # imdb['set']==1: 200, imdb['set']==2: 0, imdb['set']==3: 100,
        for i in range(train_nsamples):
            c_row = non_zero_row[shuffle_number[i]]
            c_col = non_zero_col[shuffle_number[i]]
            yy = image[c_row - HalfWidth: c_row + HalfWidth,
                       c_col - HalfWidth: c_col + HalfWidth, :]
            if args.cutout:
                yy = cutout(yy, args.cutout_length, args.num_cut)

            imdb['data'][:, :, :, i] = yy
            imdb['Labels'][i] = label[c_row, c_col].astype(np.int64)

        for i in range(validation_nsamples):
            c_row = non_zero_row[shuffle_number[i + train_nsamples]]
            c_col = non_zero_col[shuffle_number[i + train_nsamples]]
            imdb['data'][:, :, :, i + train_nsamples] = image[c_row - HalfWidth:c_row + HalfWidth,
                                                              c_col - HalfWidth:c_col + HalfWidth, :]
            imdb['Labels'][i + train_nsamples] = label[c_row, c_col].astype(np.int64)
        imdb['Labels'] = imdb['Labels'] - 1
        # 在网上查找的结果是：当有N类时，标签必须是0~(N-1)，而不能是1~N！
        # 否则会报错RuntimeError: cuda runtime error (710) : device-side assert triggered at

        train_dataset = utils.MatCifar(imdb, train=True, d=3, medicinal=0)
        valid_dataset = utils.MatCifar(imdb, train=False, d=3, medicinal=0)
        # 数据维度变化：(32, 32, 103, 200) → (200, 103, 32, 32)
        train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=0)
        valid_queue = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=0)

        # training      tar:target  pre:predict
        train_acc, train_obj, tar, pre = train(train_queue, model, criterion, optimizer)
        # validation
        valid_acc, valid_obj, tar_v, pre_v = infer(valid_queue, model, criterion)
        scheduler.step()
        toc = time.time()

        logging.info('Epoch: %03d train_loss: %f train_acc: %f valid_loss: %f valid_acc: %f' % (
            epoch + 1, train_obj, train_acc, valid_obj, valid_acc))

        if epoch > args.epochs * 0.8 and valid_obj < min_val_obj:
            min_val_obj = valid_obj
            utils.save(model, './result/weights.pt')

    # test
    # matrix = test_model(model, numbatch2, seed)
    utils.load(model, './result/weights.pt')
    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)
    for i in range(0, numbatch2, 1):            # range(0, numbatch2, numbatch2-1): 用于测试，只执行两次循环
        print('test batch: %d/%d' %(i+1, numbatch2))
        imdb = {'data': np.zeros([windowsize, windowsize, nBand, batchnumber], dtype=np.float32),
                'Labels': np.zeros([batchnumber], dtype=np.int64),
                'set': 3 * np.ones([batchnumber], dtype=np.int64)}
        for j in range(batchnumber):
            c_row = non_zero_row[shuffle_number[j + train_nsamples + validation_nsamples + i * batchnumber]]
            c_col = non_zero_col[shuffle_number[j + train_nsamples + validation_nsamples + i * batchnumber]]
            imdb['data'][:, :, :, j] = image[c_row - HalfWidth:c_row + HalfWidth,
                                             c_col - HalfWidth:c_col + HalfWidth, :]
            imdb['Labels'][j] = label[c_row, c_col].astype(np.int64)

        imdb['Labels'] = imdb['Labels'] - 1

        test_dataset = utils.MatCifar(imdb, train=False, d=3, medicinal=0)

        test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=50,
                                                 shuffle=False, num_workers=0)

        valid_acc, valid_obj, tar_v, pre_v = infer(test_queue, model, criterion)

        predict = np.append(predict, pre_v)
        labels = np.append(labels, tar_v)

        # 将剩余的不足 batchnumber=1000个的样本也用于测试集
        if i == numbatch2-1:
            rest_nsamples = test_nsamples - numbatch2 * batchnumber
            imdb = {'data': np.zeros([windowsize, windowsize, nBand, rest_nsamples], dtype=np.float32),
                    'Labels': np.zeros([rest_nsamples], dtype=np.int64),
                    'set': 3 * np.ones([rest_nsamples], dtype=np.int64)}
            for j in range(rest_nsamples):
                c_row = non_zero_row[shuffle_number[j + train_nsamples + validation_nsamples + numbatch2 * batchnumber]]
                c_col = non_zero_col[shuffle_number[j + train_nsamples + validation_nsamples + numbatch2 * batchnumber]]
                imdb['data'][:, :, :, j] = image[c_row - HalfWidth:c_row + HalfWidth,
                                                 c_col - HalfWidth:c_col + HalfWidth, :]
                imdb['Labels'][j] = label[c_row, c_col].astype(np.int64)

            imdb['Labels'] = imdb['Labels'] - 1

            test_dataset = utils.MatCifar(imdb, train=False, d=3, medicinal=0)

            test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=50,
                                                     shuffle=False, num_workers=0)

            valid_acc, valid_obj, tar_v, pre_v = infer(test_queue, model, criterion)

            predict = np.append(predict, pre_v)
            labels = np.append(labels, tar_v)
    # 未使用零头测试样本时： predict{ndarray}: .shape:(35000,) .dtype:float64, labels{ndarray}: .shape:(35000,) .dtype:float64
    # 将零头测试样本也用上后： predict{ndarray}: .shape:(35435,) .dtype:float64, labels{ndarray}: .shape:(35435,) .dtype:float64
    OA_V = sum(map(lambda x, y: 1 if x == y else 0, predict, labels)) / test_nsamples
    matrix = confusion_matrix(labels, predict)

    logging.info('test_acc= %f' % (OA_V))
    return matrix


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    tar = np.array([])
    pre = np.array([])

    for step, (input, target) in enumerate(train_queue):
        # global device
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()    # 等价于 self.train(mode=False)
    tar = np.array([])
    pre = np.array([])
    # global device
    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)

        logits, logits_aux = model(input)
        loss = criterion(logits, target)

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    # AA = np.zeros([shape[0]], dtype=np.float)
    TPR: ndarray = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        TPR[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA = np.mean(TPR)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA, Kappa, TPR


if __name__ == '__main__':
    genotype = eval('genotypes.{}'.format(args.arch))
    matrix = main(genotype=genotype, seed=np.random.randint(low=0, high=10000, size=1), cut=False)
    OA, AA, Kappa, TPR = cal_results(matrix)
    print(OA)

    # 将分类结果按列保存到txt中
    now = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
    resultStr = '# OA, AA, Kappa, TPR  ' + now + image_file + '\n'

    # 不同数据集的分类结果保存到不同的文件中，fname
    dir_name, base_name = osp.split(image_file)
    f_name, f_ext = osp.splitext(base_name)
    fname = './result/classification_3D_' + f_name + '.txt'

    a = list([OA, AA, Kappa])
    a.extend(list(TPR))
    # X = np.array(a)
    # 使用f和np.savetxt()组合来保存数据，格式处理很方便
    # 数据的标题只写一次，只要不是空文件，追加的时候就不用再写标题了
    # 数据文件fname不存在的话，open()就会自动创建fname文件
    with open(fname, mode='a', encoding='utf-8') as f:  # 'utf-8'可确保中文不乱码;如果fname不存在，则会创建fname文件
        if os.path.getsize(fname) == 0:
            header = resultStr.rstrip('\n')
        else:
            header = ''
        np.savetxt(f, np.column_stack(a), fmt='%.4f', delimiter=' ', newline='\n', header=header,
                   footer='', comments='', encoding=None)
        f.close()

    # OA, AA_mean, Kappa, AA = cal_results(matrix)
    # print(OA)
