import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from copy import deepcopy

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def clone_module_to_modulelist(module, module_num):
    """
    克隆n个Module类放入ModuleList中，并返回ModuleList，这个ModuleList中的每个Module都是一模一样的
    nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
    你可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，
    加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，
    同时 module 的 parameters 也会自动添加到整个网络中。
    :param module: 被克隆的module
    :param module_num: 被克隆的module数
    :return: 装有module_num个相同module的ModuleList
    """
    return nn.ModuleList([deepcopy(module) for _ in range(module_num)])

##----------------------------------Transformer模块------------------------------------##

class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):
        """
        :param feature: self-attention 的 x 的大小，就是d_model的维度
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # x-mean  (2,4,62)    -spatial        2,30,22
        # a_2       (62,)         spatial          22,

class SublayerConnection(nn.Module):
    """
    这不仅仅做了残差，这是把残差和 layernorm 一起给做了

    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        # 第一步做 layernorm
        self.layer_norm = LayerNorm(size)
        # 第二步做 dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        :param x: 就是self-attention的输入
        :param sublayer: self-attention层
        :return:
        """
        return self.dropout(self.layer_norm(x + sublayer(x)))


# def self_attention(query, key, value, dropout=None, mask=None):
#     """
#     自注意力计算
#     :param query: Q
#     :param key: K
#     :param value: V
#     :param dropout: drop比率
#     :param mask: 是否mask
#     :return: 经自注意力机制计算后的值
#     """
#     d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
#     # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
#     # 判断是否要mask，注：mask的操作在QK之后，softmax之前
#     if mask is not None:
#         """
#         scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
#         在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
#         """
#         # mask.cuda()
#         # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引
#
#         scores = scores.masked_fill(mask == 0, -1e9)
#
#     self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
#     # 判断是否要对相似概率分布进行dropout操作
#     if dropout is not None:
#         self_attn_softmax = dropout(self_attn_softmax)
#
#     # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
#     return torch.matmul(self_attn_softmax, value), self_attn_softmax

def self_attention(query, key, value, dropout=None, mask=None):
    """
    自注意力计算
    :param query: Q
    :param key: K
    :param value: V
    :param dropout: drop比率
    :param mask: 是否mask
    :return: 经自注意力机制计算后的值
    """

    d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
    # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算

    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    if mask is not None:
        """
        scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
        在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
        """
        # mask.cuda()
        # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引

        scores = scores.masked_fill(mask == 0, -1e9)

    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    # 判断是否要对相似概率分布进行dropout操作
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)
        attention_output = torch.matmul(self_attn_softmax, value)
    # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
    return attention_output, self_attn_softmax

class MultiHeadAttention(nn.Module):
    """
    多头注意力计算
    """

    def __init__(self, head, d_model, dropout=0.1):
        """
        :param head: 头数
        :param d_model: 词向量的维度，必须是head的整数倍
        :param dropout: drop比率
        """
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)  # 确保词向量维度是头数的整数倍
        self.d_k = d_model // head  # 被拆分为多头后的某一头词向量的维度
        self.head = head
        self.d_model = d_model

        """
        由于多头注意力机制是针对多组Q、K、V，因此有了下面这四行代码，具体作用是，
        针对未来每一次输入的Q、K、V，都给予参数进行构建
        其中linear_out是针对多头汇总时给予的参数
        """
        self.linear_query = nn.Linear(d_model, d_model)  # 进行一个普通的全连接层变化，但不修改维度
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None  # attn_softmax是能量分数, 即句子中某一个词与所有词的相关性分数， softmax(QK^T)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            """
            多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第二维（head维）添加一维，与后面的self_attention计算维度一样
            具体点将，就是：
            因为mask的作用是未来传入self_attention这个函数的时候，作为masked_fill需要mask哪些信息的依据
            针对多head的数据，Q、K、V的形状维度中，只有head是通过view计算出来的，是多余的，为了保证mask和
            view变换之后的Q、K、V的形状一直，mask就得在head这个维度添加一个维度出来，进而做到对正确信息的mask
            """
            mask = mask.unsqueeze(1)

        n_batch = query.size(0)  # batch_size大小，假设query的维度是：[10, 32, 512]，其中10是batch_size的大小

        """
        下列三行代码都在做类似的事情，对Q、K、V三个矩阵做处理
        其中view函数是对Linear层的输出做一个形状的重构，其中-1是自适应（自主计算）
        从这种重构中，可以看出，虽然增加了头数，但是数据的总维度是没有变化的，也就是说多头是对数据内部进行了一次拆分
        transopose(1,2)是对前形状的两个维度(索引从0开始)做一个交换，例如(2,3,4,5)会变成(2,4,3,5)
        因此通过transpose可以让view的第二维度参数变成n_head
        假设Linear成的输出维度是：[10, 32, 512]，其中10是batch_size的大小
        注：这里解释了为什么d_model // head == d_k，如若不是，则view函数做形状重构的时候会出现异常
        """

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]，head=8
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]


        # x是通过自注意力机制计算出来的值， self.attn_softmax是相似概率分布
        x, self.attn_softmax = self_attention(query, key, value, dropout=self.dropout, mask=mask)

        """
        下面的代码是汇总各个头的信息，拼接后形成一个新的x
        其中self.head * self.d_k，可以看出x的形状是按照head数拼接成了一个大矩阵，然后输入到linear_out层添加参数
        contiguous()是重新开辟一块内存后存储x，然后才可以使用.view方法，否则直接使用.view方法会报错
        """
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)


class FeedForward(nn.Module):
    """
    两层具有残差网络的前馈神经网络，FNN网络
    """

    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        """
        :param d_model: FFN第一层输入的维度
        :param d_ff: FNN第二层隐藏层输入的维度
        :param dropout: drop比率
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: 输入数据，形状为(batch_size, input_len, model_dim)
        :return: 输出数据（FloatTensor），形状为(batch_size, input_len, model_dim)
        """
        inter = self.dropout_1(self.elu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        # return output + x   # 即为残差网络
        return output  # + x

class SublayerConnection(nn.Module):
    """
    子层的连接: layer_norm(x + sublayer(x))
    上述可以理解为一个残差网络加上一个LayerNorm归一化
    """

    def __init__(self, size, dropout=0.1):
        """
        :param size: d_model
        :param dropout: drop比率
        """
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        # TODO：在SublayerConnection中LayerNorm可以换成nn.BatchNorm2d
        # self.layer_norm = nn.BatchNorm2d()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(self.layer_norm(x + sublayer(x)))

class EncoderLayer(nn.Module):
    """
    一层编码Encoder层
    MultiHeadAttention -> Add & Norm -> Feed Forward -> Add & Norm
    """

    def __init__(self, size, attn, feed_forward, dropout=0.1):
        """
        :param size: d_model
        :param attn: 已经初始化的Multi-Head Attention层
        :param mode: Attention模块的计算模式
        :param feed_forward: 已经初始化的Feed Forward层
        :param dropout: drop比率
        """
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        """
        下面一行的作用是因为一个Encoder层具有两个残差结构的网络
        因此构建一个ModuleList存储两个SublayerConnection，以便未来对数据进行残差处理
        """
        self.sublayer_connection_list = clone_module_to_modulelist(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """
        :param x: Encoder层的输入
        :param mask: mask标志
        :return: 经过一个Encoder层处理后的输出
        """
        """
        编码层第一层子层
        self.attn 应该是一个已经初始化的Multi-Head Attention层
        把Encoder的输入数据x和经过一个Multi-Head Attention处理后的x_attn送入第一个残差网络进行处理得到first_x
        """
        first_x = self.sublayer_connection_list[0](x, lambda x_attn: self.attn(x, x, x, mask))

        """
        编码层第二层子层
        把经过第一层子层处理后的数据first_x与前馈神经网络送入第二个残差网络进行处理得到Encoder层的输出
        """
        return self.sublayer_connection_list[1](first_x, self.feed_forward)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=22):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class Encoder(nn.Module):
    """
    构建n层编码层
    """

    def __init__(self, n, encoder_layer):
        """
        :param n: Encoder层的层数
        :param encoder_layer: 初始化的Encoder层
        """
        super(Encoder, self).__init__()
        self.encoder_layer_list = clone_module_to_modulelist(encoder_layer, n)

    def forward(self, x, src_mask):
        """
        :param x: 输入数据
        :param src_mask: mask标志
        :return: 经过n层Encoder处理后的数据
        """
        for encoder_layer in self.encoder_layer_list:
            x = encoder_layer(x, src_mask)
        return x


##########################################新的模型##########################################

class S_Backbone_test(nn.Module):
    def __init__(self, kernel_length, dropout):
        super(S_Backbone_test, self).__init__()
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, kernel_length), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 4))
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(kernel_length, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 4))

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(1,4), stride=(1,10))

    def forward(self, x):
        x = x.unsqueeze(1)  # bathc, 1, ch, 400
        # Layer 1
        x = self.conv1(x)  # batch, 40, 5, 396
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        # x = x.squeeze(2)
        x = self.conv3(x)

        # # FC Layer
        x = x.reshape(-1, 20 * 2)   # 这个卷积的shape也可以调整
        return x

class T_Backbone_test(nn.Module):
    def __init__(self, kernel_length, dropout):
        super(T_Backbone_test, self).__init__()
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, kernel_length), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 4))
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(kernel_length, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 4))

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(1, 4), stride=(1, 8))

    def forward(self, x):
        x = x.unsqueeze(1)  # bathc, 1, ch, 400
        # Layer 1
        x = self.conv1(x)  # batch, 40, 5, 396
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)   # 这个卷积的shape也可以调整

        # # FC Layer
        x = x.reshape(-1, 20*2)
        return x


class S_Backbone(nn.Module):
    def __init__(self, kernel_length, dropout):
        super(S_Backbone, self).__init__()

        self.kernel_length = kernel_length
        self.dropout = dropout
        # Layer 1
        # self.conv1 = nn.Conv2d(1, 16, (1, self.kernel_length), padding=(0,  self.kernel_length//2))
        self.conv1 = nn.Conv2d(1, 16, (1, self.kernel_length), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 64))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 32))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 16, 20)

    def forward(self, x):
        x = x.unsqueeze(1)    # (2,1,20,400)
        x = x.permute(0, 1, 3, 2) # (sample,1,400,20)
        # Layer 1
        x = F.elu(self.conv1(x))  # (sample,16,400,1)
        x = self.batchnorm1(x)
        x = F.dropout(x, self.dropout)
        # Transpose
        x = x.permute(0, 3, 1, 2)   # (sample,1,16,400)

        # Layer 2
        x = self.padding1(x)        # (sample,1,17,433)
        x = F.elu(self.conv2(x))    # (sample,4,16,370)
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling3(x)

        # # FC Layer
        x = x.reshape(-1, 4 * 2 * 16)
        x = F.sigmoid(self.fc1(x))
        return x

class T_Backbone(nn.Module):
    def __init__(self, ch, dropout):
        super(T_Backbone, self).__init__()

        self.kernel_length = ch
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=(1, self.kernel_length), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=4,
                               kernel_size=(2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=4,
                               kernel_size=(8, 16))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 3, 20)

    def forward(self, x):
        x = x.unsqueeze(1)    # (2,1,20,400)
        x = x.permute(0, 1, 3, 2) # (sample,1,400,20)
        # Layer 1
        x = F.elu(self.conv1(x))  # (sample,16,400,1)
        x = self.batchnorm1(x)
        x = F.dropout(x, self.dropout)
        # Transpose
        x = x.permute(0, 3, 1, 2)   # (sample,1,16,400)

        # Layer 2
        x = self.padding1(x)        # (sample,1,17,433)
        x = F.elu(self.conv2(x))    # (sample,4,16,370)
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling3(x)

        # # FC Layer
        x = x.reshape(-1, 4 * 2 * 3)
        x = F.sigmoid(self.fc1(x))
        return x


class Spatial_Local_Conv(nn.Module):
    def __init__(self, division, dropout):
        super(Spatial_Local_Conv, self).__init__()
        self.division = division
        self.dropout = dropout

        for i in self.division.keys():
            rigion_conv = S_Backbone_test(len(self.division[i]), dropout)
            setattr(self, f'S_region_conv{i}', rigion_conv)

    def forward(self, x):
        encoded_regions = []

        # 根据给定的region进行channel维度的切分
        for i in self.division.keys():
            region = x[:, self.division[i], :]
            encoded_region = getattr(self, f'S_region_conv{i}')(region)
            encoded_region = encoded_region.unsqueeze(1)
            encoded_regions.append(encoded_region)
        return encoded_regions

class Temporal_Local_Conv(nn.Module):
    def __init__(self, division, dropout):
        super(Temporal_Local_Conv, self).__init__()
        self.division = division
        self.dropout = dropout

        for i in self.division.keys():
            rigion_conv = T_Backbone_test(30, dropout)
            setattr(self, f'T_region_conv{i}', rigion_conv)

    def forward(self, x):
        encoded_regions = []

        # 根据给定的region进行channel维度的切分
        for i in self.division.keys():
            region = x[:, :, self.division[i]]
            encoded_region = getattr(self, f'T_region_conv{i}')(region)
            encoded_region = encoded_region.unsqueeze(1)
            encoded_regions.append(encoded_region)
        return encoded_regions


class Local_Encoder(nn.Module):
    def __init__(self, spatial_div_dict, temporal_div_dict, d_model_dic, head_dic, d_ff, n_layers, dropout):
        super(Local_Encoder, self).__init__()
        self.d_model_dic = d_model_dic
        # 空间维度backbone分块卷积
        self.Local_spatial_conved = Spatial_Local_Conv(spatial_div_dict, dropout)
        self.Local_temporal_conved = Temporal_Local_Conv(temporal_div_dict, dropout)

        # 空间区域的Attention
        s_region_att = MultiHeadAttention(head_dic['spatial'], d_model_dic['spatial'], dropout)
        s_feed_forward = FeedForward(d_model_dic['spatial'], d_ff)
        self.s_encoder_region = Encoder(n_layers, EncoderLayer(d_model_dic['spatial'],
                                                          deepcopy(s_region_att), deepcopy(s_feed_forward)))

        # 空间区域的Attention
        t_region_att = MultiHeadAttention(head_dic['temporal'], d_model_dic['temporal'], dropout)
        t_feed_forward = FeedForward(d_model_dic['temporal'], d_ff)
        self.t_encoder_region = Encoder(n_layers, EncoderLayer(d_model_dic['temporal'],
                                                               deepcopy(t_region_att), deepcopy(t_feed_forward)))

        # 时空特征区域特征融合fusion模块
        self.st_fusiondiv = len(spatial_div_dict) + len(temporal_div_dict)
        st_fusion_att = MultiHeadAttention(head_dic['st_fusion'], d_model_dic['st_fusion'], dropout)
        st_fusion_ff = FeedForward(d_model_dic['st_fusion'], d_ff)
        self.st_fusion_encoder = Encoder(n_layers, EncoderLayer(d_model_dic['st_fusion'],
                                                                deepcopy(st_fusion_att), deepcopy(st_fusion_ff)))



    def forward(self, x):
        temp1 = self.Local_spatial_conved(x)
        temp2 = self.Local_temporal_conved(x)
        S_Region_tensor = torch.cat(temp1, dim=1)
        T_Region_tensor = torch.cat(temp2, dim=1)
        Spatial_encoding = self.s_encoder_region(S_Region_tensor, None)
        Temporal_encoding = self.t_encoder_region(T_Region_tensor, None)
        st_encoding = torch.cat([Spatial_encoding, Temporal_encoding], dim=1)
        # ret = (0.8 * Spatial_encoding.mean(1) + 0.2 *Temporal_encoding.mean(1)) / 2
        ret = self.st_fusion_encoder(st_encoding, None)
        ret = ret.reshape(-1, self.st_fusiondiv * self.d_model_dic['st_fusion'])
        return  ret





##----------------------------- 模型整体骨干-----------------------------##
class L2GNet(nn.Module):
    def  __init__(self, spatial_div_dict, temporal_div_dict ,d_model_dic,  head_dic, d_ff, n_layers, dropout,
                  clf_class=4, domain_class=8):
        super(L2GNet, self).__init__()
        self.linear_dim = (len(spatial_div_dict) + len(temporal_div_dict)) * d_model_dic['st_fusion']
        ## Spatial-Temporal Local to Global ##
        self.L2G = nn.Sequential()
        self.L2G.add_module('L2G', Local_Encoder(spatial_div_dict, temporal_div_dict, d_model_dic,
                                                            head_dic, d_ff, n_layers, dropout))

        # Class classifier layer
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.linear_dim, 16))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(16))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(dropout))
        self.class_classifier.add_module('c_fc2', nn.Linear(16, clf_class))
        #
        # Domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.linear_dim, 16))
        # self.domain_classifier.add_module('d_fc1', nn.Linear(4840, 32))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(16))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_drop1', nn.Dropout(dropout))
        self.domain_classifier.add_module('d_fc2', nn.Linear(16, domain_class))


    def forward(self, x, alpha):
        x = self.L2G(x)  # 2, ch, 62
        # feature = self.temporal_L_to_G(x)
        ### channel 维度的加权 to-Global
        # [2,22,24]
        # feature = x.view(-1,x.shape[1] * x.shape[2])  # 将维度拉平
        # feature = torch.mean(x, 1)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        class_output = self.class_classifier(x)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output



if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 30, 400))
    s_division = {
        '1':[i for i in range(5)],
        '2':[i for i in range(5,15)],
        '3':[i for i in range(15,30)],
    }
    t_divison = {
        '1':[i for i in range(100)],
        '2':[i for i in range(100,200)],
        '3':[i for i in range(200,300)],
        '4':[i for i in range(300,400)],
    }
    d_model_dic = {
        'spatial':40,
        'temporal':40,
        'st':40
    }
    head_dic = {
        'spatial': 1,
        'temporal': 1,
        'st_fusion':1

    }
    d_ff = 2
    n_layers = 2
    dropout = 0.3
    # model = Local_Encoder(s_division, t_divison, d_model_dic, head_dic, d_ff, n_layers, dropout)
    model = L2GNet(s_division, t_divison ,d_model_dic,  head_dic, d_ff, n_layers, dropout,
                  clf_class=4, domain_class=8)
    res = model(inp, 0.1)

