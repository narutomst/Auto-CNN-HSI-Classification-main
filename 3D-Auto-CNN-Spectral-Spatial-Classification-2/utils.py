import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
import torch.utils.data as data


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))  # 原版
        mask = torch.ones(x.size(0), 1, 1, 1, device=device).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def cutout(img, length, num_band):
    img = np.transpose(img, (2, 0, 1))

    c, h, w = np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]

    data = img
    RandPerm = np.random.permutation(c)
    for i in range(len(RandPerm) // num_band):
        img_c = img[RandPerm[i], :, :]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0

        img_c *= mask
        img_c = img_c[np.newaxis, :, :]
        data[RandPerm[i], :, :] = img_c

    img = np.transpose(data, (1, 2, 0))

    return img


class MatCifar(torch.utils.data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb    # {dict}{'data':,'Labels':,'set':}
        self.d = d
        train_logical_index = self.imdb['set'] == 1
        test_logical_index = self.imdb['set'] == 3
        self.train_labels = self.imdb['Labels'][train_logical_index]  # 使用逻辑索引 logical indexing
        self.test_labels = self.imdb['Labels'][test_logical_index]
        if medicinal == 4 and d == 2:
            self.train_data = self.imdb['data'][train_logical_index, :]
            self.test_data = self.imdb['data'][test_logical_index, :]

        if medicinal == 1:
            self.train_data = self.imdb['data'][train_logical_index, :, :, :]
            self.test_data = self.imdb['data'][test_logical_index, :, :, :]

        else:
            self.train_data = self.imdb['data'][:, :, :, train_logical_index]  # imdb['data'].shape: (32, 32, 103, 200);
            self.test_data = self.imdb['data'][:, :, :, test_logical_index]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))
                # 维度变化：(32, 32, 103, 200) → (200, 103, 32, 32)
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))
        # # 判断两个ndarray是否完全相同，可以用如下的方法
        # a = self.imdb['data'][:, :, :, self.imdb['set'] == 1]   # 逻辑索引 logical indexing
        # print(np.array_equal(self.train_data, a))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
