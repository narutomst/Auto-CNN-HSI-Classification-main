import torch
import numpy as np
# import torch.nn as nn


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])
    # view()函数的作用：把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），
    # 然后按照形状参数组合成其他形状的tensor
    # x.view(-1)中的-1,view()的作用和reshape类似，-1通常是根据另外几个形状参数计算出来的。
    # 但是这里只有一个维度，因此就会将x里面的所有维度数据按照行优先的顺序排成一个一维的数据。
    # 列表推导式的写法
    # torch.cat(tensors, dim=0, *, out=None) → Tensor
    # 将给定顺序的seq张量序列，按照指定的轴连接起来。dim=0，表示在shape[0]维度上连接；dim=1，表示在shape[1]维度上连接.
    # 注意：对于tensor而言，shape 是 tensor 的属性；size() 是tensor 的方法；但返回结果都是一样的，比如torch.Size([3, 4])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        # eta: 即学习率lr, float 0.004
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        # _concat()定义在 line 7
        # self.model.parameters()： <generator object Module.parameters at 0x000002038E2C1148>
        # _concat(self.model.parameters())： tensor([ 0.0580,  0.0978,  0.0657,  ..., -0.0783, -0.0189,  0.0881], device='cuda:0', grad_fn=<CatBackward>)
        # _concat(self.model.parameters()).data:
        # tensor([ 0.0580,  0.0978,  0.0657,  ..., -0.0783, -0.0189,  0.0881], device='cuda:0')
        # _concat(self.model.parameters()).data.shape: torch.Size([157817])
        # theta的含义是什么？

        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
            # _concat()返回的是一个一维的tensor
            # TORCH.TENSOR.MUL_即tensor.mul_()，是支持broadcast特性的乘法运算
            # self.network_momentum: {float} 0.9
            # 整个表达式的含义是：moment = _concat()*0.9

            # 列表推导式的写法
            # network_optimizer: Adam (
            # Parameter Group 0
            #     amsgrad: False
            #     betas: (0.9, 0.999)
            #     eps: 1e-08
            #     initial_lr: 0.004
            #     lr: 0.004
            #     weight_decay: 0.0003
            # )
        except:
            moment = torch.zeros_like(theta)  # moment:{tensor}, moment.shape: torch.Size([157817])

        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        # torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
        # 计算并返回输出outputs相对于输入inputs的梯度之和。这里的outputs的requires_grad属性必须为True
        # weight decay是一种神经网络regularization的方法，它的作用在于让weight不要那么大，实践的结果是这样做可以有效防止overfitting。
        # weight decay也被称为L2 regularization，或者是L2 parameter norm penalty
        # self.network_weight_decay：{float} 0.0003, theta:{tensor}, theta.shape: torch.Size([157817])
        # unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))  # 原版sub()函数报错
        # unrolled_model = self._construct_model_from_theta(theta.sub(eta * (moment + dtheta)))  # 按照自己的理解修改，正确
        # unrolled_model = self._construct_model_from_theta(theta.sub((moment + dtheta), eta))
        # 调换参数顺序，继续会报错。sub() takes 1 positional argument but 2 were given
        # 由此想到此种情况下只需要一个位置参数，那么标准用法：torch.sub(input, other, *, alpha=1, out=None) → Tensor的说明里，
        # input是self，即theta自身；other就是sub()所需的唯一一个位置参数；*不代表实际参数，仅是一个标志：
        # 即，从星号*后面的参数应该都是关键字参数，必须以显示的形式给sub()函数传值
        unrolled_model = self._construct_model_from_theta(theta.sub((moment + dtheta), alpha=eta))  # 运行正确
        # unrolled_model = self._construct_model_from_theta(theta.sub(alpha=eta, (moment + dtheta)))    #报错，位置参数的位置不能被关键字参数占了
        # theta: tensor([0.0915, 0.0792, 0.0122,  ..., 0.0594, 0.0200, 0.0714], device='cuda:0')
        # eta: 即学习率lr, float 0.004
        # moment: tensor([0., 0., 0.,  ..., 0., 0., 0.], device='cuda:0')
        # dtheta: tensor([-0.0166, -0.0151, -0.0139,  ...,  0.0564, -0.1387,  0.1128], device='cuda:0')
        # dtheta.shape: torch.Size([157817])
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):  # eta即学习率lr
        self.optimizer.zero_grad()
        if unrolled:  # eta即学习率lr
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        # eta即学习率lr
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            # g.data.sub_(eta, ig.data)  # 原版报错
            # g.data: {tensor}, .shape: torch.Size([14,8]), .dtype: torch.float32
            # eta: {float} 0.004
            # ig.data: {tensor}, v.shape: torch.Size([14,8]), dtype: torch.float32
            # 根据以上各个参数形状特点以及add()函数的用法，修改为如下
            g.data.sub_(ig.data, alpha=eta)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data.to(device)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            # p.data.add_(R, v)  # 原版报错
            # p.data: {tensor}, .shape: torch.Size([9]), .dtype: torch.float32
            # R: tensor(0.0018, device='cuda:0')
            # v: {tensor}, v.shape: torch.Size([9]), dtype: torch.float32
            # 根据以上各个参数形状特点以及add()函数的用法，修改为如下
            p.data.add_(v, alpha=R)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # p.data.sub_(2 * R, v)  # 原版报错
            # p.data: {tensor}, .torch.Size([32, 102, 1, 1]), .dtype: torch.float32
            # R: tensor(0.0025, device='cuda:0')
            # v: {tensor}, .shape: torch.Size([32, 102, 1, 1]), .dtype: torch.float32
            # 根据以上各个参数形状特点以及sub()函数的用法，修改为如下
            p.data.sub_(v, alpha=2 * R)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # p.data.add_(R, v)  # 原版报错
            # p.data: {tensor}, .shape: torch.Size([32, 102, 1, 1]), .dtype: torch.float32
            # R: tensor(0.0022, device='cuda:0')
            # v: {tensor}, .shape: torch.Size([32, 102, 1, 1]), dtype: torch.float32
            # 根据以上各个参数形状特点以及add()函数的用法，修改为如下
            p.data.add_(v, alpha=R)
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
        # div()是绝对不会出错的，因为只需要一个位置参数，再不需要其他参数
