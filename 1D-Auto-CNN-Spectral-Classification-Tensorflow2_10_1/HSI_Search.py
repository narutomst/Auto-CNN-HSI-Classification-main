import os
import argparse
import logging
import random
import sys
import time  # 判断time模块是否导入成功，'time' in sys.modules.keys()：True，说明该模块导入成功
import numpy as np
import torch  # 运行前，'torch' in sys.modules.keys()：False; 运行后，'torch' in sys.modules.keys()：True
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils

# 如果前面的某个模块内第一句经执行了 import global_variable as glv，下面一句是多余的了吗？
# 不是的！为了显式使用glv，下面一句必不可少
# 因为上面一句的执行结果是：'global_variable' in sys.modules.keys()：True;然而 'glv' in sys.modules.keys()：False
import global_variable as glv
import utils
from architect import Architect
from data_prepare import read_data
from model_search import Network

# 本句执行完以后，结果还是：'global_variable' in sys.modules.keys()：True;然而 'glv' in sys.modules.keys()：False
# 所以，检测某个模块是否导入成功，一定要用该模块的完整名称，而不能是 as 后面的简略名称

glv._init()  # 先必须在主模块初始化一次（且只需要在Main模块一次即可）

# glv.set_value('num_bands', data[0].images.shape[1])

# 然后其他的任何*.py文件只需要导入即可使用：
# import global_variable as glv
# #不需要再初始化了
# ROOT = glv.set_value('ROOT',80)
# CODE = glv.get_value('CODE')
# num_bands = glv.get_value('num_bands')

# argsparse是python的命令行解析的标准模块，内置于python，不需要安装。
# 这个库可以让我们直接在命令行中就可以向程序中传入参数并让程序运行。
# argparse --- 命令行选项、参数和子命令解析器
# argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，
# 然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
# argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

# 创建一个解析器
parser = argparse.ArgumentParser("HSI")
# ArgumentParser(prog='HSI', usage=None, description=None, formatter_class=<class 'argparse.HelpFormatter'>,
# conflict_handler='error', add_help=True)
# 添加参数
parser.add_argument('--num_class', type=int, default=9, help='classes of HSI dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=4e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=3e-4, help='weight decay for arch encoding')
# 解析参数
args = parser.parse_args()

# 日志消息将分为两路输出，一路是console控制台，另一路是日志文件log.txt
# 要实现以上两个功能，需要分别向根记录器（root Logger）添加(.addHandler)两个处理器
# StreamHandler和FileHandler(注意：FileHandler继承自StreamHandler,而StreamHandler继承自Handler)
# 定义在https://github.com/python/cpython/blob/3.9/Lib/logging/__init__.py

# 通过logging.basicConfig函数对其中一路日志的输出格式及方式做相关配置
# 这其实是basicConfig()通过为根记录器（root Logger）创建并添加某个特定处理器来实现的
# 只不过这不是为记录器对象Logger创建和添加处理器的标准做法
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')  # '%m/%d %I:%M:%S %p'
# 为根记录器创建StreamHandler处理器对象（并直接将其添加到根记录器），
# 消息输出指定为：在标准输出设备sys.stdout上（默认为sys.stderr），
# 消息级别指定为：INFO及以上(Handler及其子类的默认消息级别level=NOTSET)，
# 请注意，根记录器(root Logger)的默认级别为 `WARNING`消息格式为log_format
# 日期格式datefmt指定为与 time.strftime() 一致。：https://docs.python.org/zh-cn/3/library/time.html#time.strftime
# %m：十进制数 [01,12] 表示的月。
# %d：十进制数 [01,31] 表示的月中日。
# %I：十进制数 [01,12] 表示的小时（12小时制）。
# %M：十进制数 [00,59] 表示的分钟。
# %S：十进制数 [00,61] 表示的秒。
# %p：本地化的 AM 或 PM 。
# 日期时间的输出效果  08/10 11:12:29 PM

# 也可以选择基于basicConfig()函数，首先为第二路日志做相关设置，具体语法为：
# logging.basicConfig(filename='./result/log.txt', level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
# 若是这样，后面就得通过创建StreamHandler处理器来完成第一路输出的任务了。因为在basicConfig()中，stream参数和filename参数
# 不能同时出现，否则将会报错

# Logger 对象可以使用 addHandler() 方法向自己添加零个或多个处理器对象以完成多路输出。
# 作为示例场景，应用程序可能希望将所有日志消息（level=DEBUG）发送到日志文件(log.txt)，
# 将错误（level=ERROR）或更高的所有日志消息发送到标准输出(sys.stdout或sys.stderr)，
# 以及将所有关键消（level=WARNING）息发送至一个邮件地址。
# 此方案需要三个单独的处理器(一个FileHandler,两个StreamHandler)，其中每个处理器负责将特定严重性(level)的消息发送到特定位置。

fh = logging.FileHandler(filename='./result/log.txt')
# 创建FileHandler处理器对象，消息输出到本地文件'./result/log.txt'中.
# logging.FileHandler(filename, mode='a', encoding=None, delay=False, errors=None)
# Handler 对象负责将适当的日志消息（基于日志消息的严重性）分派给处理器的指定目标。
# Handler及其子类创建处理器时，日志级别被设置为 NOTSET （所有的消息都会被处理）,用户可以通过类方法setLevel()来设置消息级别
#
fmt = logging.Formatter(log_format)
# class logging.Formatter(fmt=None, datefmt=None, style='%', validate=True)¶
# 返回 Formatter 类的新实例。实例将使用整个消息的格式字符串以及消息的日期/时间部分的格式字符串进行初始化。
# 如果未指定 fmt ，则使用 '%(message)s'。如果未指定 datefmt，则使用 formatTime() 文档中描述的格式。
# 构造函数有三个可选参数 —— 消息格式字符串、日期格式字符串和样式指示符。

# logging.Formatter.__init__(fmt=None, datefmt=None, style='%')
# 如果没有消息格式字符串，则默认使用原始消息。如果没有日期格式字符串，则默认日期格式为：
# %Y-%m-%d %H:%M:%S 最后加上毫秒数。  输出效果 2021-08-16 12:54:12,457
# style 是 ％，'{ ' 或 '$' 之一。 如果未指定，则将使用 '％'。
#
# 如果 style 是 '％'，则消息格式字符串使用 %(<dictionary key>)s 样式字符串替换；
# 可能的键值在 LogRecord 属性 中。 如果样式为 '{'，则假定消息格式字符串与 str.format() （使用关键字参数）兼容，
# 而如果样式为 '$' ，则消息格式字符串应符合 string.Template.substitute() 。

fh.setFormatter(fmt)
# 处理器中很少有方法可供应用程序开发人员使用。使用内置处理器对象（即不创建自定义处理器）的
# 应用程序开发人员能用到的仅有以下配置方法：
# setLevel() 方法，就像在记录器对象中一样，指定将被分派到适当目标的最低严重性。
# 为什么有两个 setLevel() 方法？记录器中设置的级别确定将传递给其处理器的消息的严重性。
# 每个处理器中设置的级别确定该处理器将发送哪些消息（使消息级别更加细化和个性化）。
# setFormatter() 将该处理器的 Formatter 对象设置为 fmt
# addFilter() 和 removeFilter() 分别在处理器上配置和取消配置过滤器对象。

logging.getLogger().addHandler(fh)
# logging.getLogger(name=None)
# 返回具有指定 name 的日志记录器，或者当 name 为 None 时返回层级结构中的根日志记录器。
# 如果指定了 name，它通常是以点号分隔的带层级结构的名称，如 'a'、'a.b' 或 'a.b.c.d'。
# 这些名称的选择完全取决于使用 logging 的开发者。
# 所有用给定的 name 对该函数的调用都将返回相同的日志记录器实例。 这意味着日志记录器实例不需要在应用的各部分间传递。
# 默认的级别是 WARNING，意味着只会追踪该级别及以上的事件，除非更改日志配置。
# 为根记录器（root Logger）添加第二个处理器FileHandler，以实现第二路输出的功能。


# nband=103, nclass=9  "C:\Matlab练习\duogun\PaviaU.mat"  原版用的输入数据是PaviaU
# nband=102, nclass=9  "C:\Matlab练习\duogun\Pavia.mat"  当前用的输入数据是Pavia
# image_file = r'C:\Matlab练习\duogun\Pavia.mat'
# label_file = r'C:\Matlab练习\duogun\Pavia_gt.mat'
image_file = r'C:\Matlab练习\duogun\PaviaU.mat'
label_file = r'C:\Matlab练习\duogun\PaviaU_gt.mat'
disk_name = ['C:\\', 'D:\\', 'E:\\', 'F:\\', 'G:\\']  # 盘符
intermediate_path = r'Matlab练习\duogun'  # 中间路径名称
mat_file_name = ['Pavia', 'PaviaU']
# 1.搜索哪个盘中有'Matlab练习\duogun'这个路径
for item in disk_name:
    test_path = os.path.join(item, intermediate_path)
    if os.path.exists(test_path):   #
        for fname in mat_file_name:
            full_path = os.path.join(test_path, fname + '.mat')   # 2.确认该文件夹中有所指定的mat数据文件
            if os.path.isfile(full_path):
                image_file = full_path
            full_path2 = os.path.join(test_path, fname + '_gt.mat')
            if os.path.isfile(full_path2):
                label_file = full_path2
                break
    else:
        continue
    break

# 在本py文件内的使用：定义跨模块全局变量，赋值
glv.set_value('image_file', image_file)
glv.set_value('label_file', label_file)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(seed):
    data, shuffle_number = read_data(image_file, label_file, train_nsamples=200, validation_nsamples=100,
                                     windowsize=1, istraining=True, rand_seed=seed)
    if not torch.cuda.is_available():
        logging.warning('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(random.randint(1, 10000))
    cudnn.enabled = True
    torch.cuda.manual_seed(random.randint(1, 10000))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # args.init_channels = data[0].images.shape[1]
    # global 字典变量赋值
    glv.set_value('num_bands', data[0].images.shape[1])  # (200, 102, 1, 1)
    model = Network(args.init_channels, args.num_class, args.layers, criterion)  # model_search.py: line 63
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.25)

    architect = Architect(model, args)

    min_valid_obj = 100

    genotype = model.genotype()  # model_search.py: line 142
    # print('genotype = ', genotype) # y
    logging.info('genotype = %s', genotype)

    for epoch in range(args.epochs):
        tic = time.time()

        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %03d lr %e', epoch + 1, lr)

        # training    tar:target  pre:predict
        train_acc, train_obj, tar, pre = train(data.train, data.validation, model, architect, criterion, optimizer, lr)

        # validation   tar_v:target_valid  pre_v:predict_valid
        valid_acc, valid_obj, tar_v, pre_v = infer(data.validation, model, criterion)
        scheduler.step()
        toc = time.time()

        logging.info('Epoch %03d: train_loss = %f, train_acc = %f, val_loss = %f, val_acc = %f, time = %f', epoch + 1,
                     train_obj, train_acc, valid_obj, valid_acc, toc - tic)

        if valid_obj < min_valid_obj:
            genotype = model.genotype()
            # print('genotype = ', genotype)
            logging.info('genotype = %s', genotype)
            min_valid_obj = valid_obj

    return genotype


def train(train_data, valid_data, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()     # objs
    top1 = utils.AvgrageMeter()     # top1
    tar = np.array([])      # tar:target  pre:predict
    pre = np.array([])

    total_batch = int(train_data.num_examples / args.batch_size)
    for i in range(total_batch):
        input, target = train_data.next_batch(args.batch_size)

        model.train()
        n = input.shape[0]

        global device
        input = torch.from_numpy(input).to(device)
        target = torch.from_numpy(np.argmax(target, axis=1)).to(device)

        input_search, target_search = valid_data.next_batch(args.batch_size)
        input_search = torch.from_numpy(input_search).to(device)
        target_search = torch.from_numpy(np.argmax(target_search, axis=1)).to(device)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))
        # objs.update(loss.data[0], n)  # 原版报错
        # objs.update(loss.data.item(), n)  # 运行正确
        objs.update(loss.item(), n)  # 运行正确
        # top1.update(prec1[0].data[0], n)  # 原版报错
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


def infer(valid_data, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()
    tar = np.array([])
    pre = np.array([])
    global device
    total_batch = valid_data.num_examples // args.batch_size
    for i in range(total_batch):
        input, target = valid_data.next_batch(args.batch_size)
        n = input.shape[0]

        input = torch.from_numpy(input).to(device)
        target = torch.from_numpy(np.argmax(target, axis=1)).to(device)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))

        # objs.update(loss.data[0], n)  # 原版报错
        # objs.update(loss.data.item(), n)  # 运行正确
        objs.update(loss.item(), n)  # 运行正确
        # top1.update(prec1[0].data[0], n)  # 原版报错
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


if __name__ == '__main__':
    genotype = main(seed=np.random.randint(low=0, high=10000, size=1))  # main(seed)定义在line 61
    str1 = 'Searched Neural Architecture:'
    print(str1)
    print(genotype)

    # 保存优化结果到日志
    logging.info(str1)
    log_format = '%(message)s'  # 优化结果中不用显示时间等无关信息
    fh.setFormatter(logging.Formatter(log_format))
    logging.info(genotype)

    # 将优化结果自动写入到genotypes.py文件的末尾
    now = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
    resultStr = '\n# ' + now + str1 + '\n' + 'HSI = ' + str(genotype) + '\n'

    with open('genotypes.py', 'a') as f:
        f.write(resultStr)
        f.close()

    # 将本次运行中所使用的全局变量: image_file, label_file, num_bands
    # 写入到 global_variable.py 的初始化函数 _init()的全局变量字典_global_dict = {}当中去
    # 这样的话，就能实现当HSI_Classification.py 单独运行时，全局变量字典_global_dict 将以
    # 最近一次所使用的值被初始化

    # 生成最新行new_line
    prefix = '    _global_dict = {'
    _global_dict = {'image_file': image_file, 'label_file': label_file, 'num_bands': glv.get_value('num_bands')}
    new_line = prefix.strip('{') + str(_global_dict)

    # 替换
    filepath = 'global_variable.py'
    with open(filepath, mode='r', encoding='utf-8') as f:  # 'utf-8'可确保中文不乱码;如果fname不存在，则会创建fname文件
        lines = f.readlines()

    with open(filepath, mode='w', encoding='utf-8') as ff:  # 此处只能是 w，以 w 模式打开文件时，文件内容全被清空
        for line in lines:
            if line.startswith(prefix):
                line = new_line
            ff.write(line)
    # ————————————————
    # 修改指定行的方法参考了
    # 原文链接：https://blog.csdn.net/qq_36072270/article/details/103496152
