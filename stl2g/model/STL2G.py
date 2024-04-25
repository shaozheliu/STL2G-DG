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



#########-----------------------模型部分------------------------###########
class Temporal_Embedding2(nn.Module):
    def __init__(self, dropout):
        super(Temporal_Embedding2, self).__init__()
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(1, 5, (1, 5), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(5, False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Layer 2
        self.conv2 = nn.Conv2d(5, 10, (1, 10))
        self.batchnorm2 = nn.BatchNorm2d(10, False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 4))

        # Layer 3
        self.conv3 = nn.Conv2d(10, 25, (1, 10))
        self.batchnorm3 = nn.BatchNorm2d(25, False)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 4))

        # Layer 4
        self.conv4 = nn.Conv2d(25, 50, (1, 5))
        self.batchnorm4 = nn.BatchNorm2d(50, False)
        self.maxpooling4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))


    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.maxpooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.maxpooling2(x)
        x = F.dropout(x, self.dropout)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.maxpooling3(x)
        x = F.dropout(x, self.dropout)
        x = torch.mean(x, dim=1)
        return x


class Temporal_Embedding(nn.Module):
    def __init__(self,in_channels, out_channels, dropout):
            super(Temporal_Embedding, self).__init__()
            self.dropout = dropout
            # Layer 1
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(1, 30), stride=(1,3), padding=(0,10))

            # Layer 2
            self.conv2 = nn.Conv2d(out_channels, out_channels, (in_channels, 1), padding=(0,5),bias=False)
            self.batchnorm = nn.BatchNorm2d(out_channels)
            self.pooling2 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 2))

            # Layer 3
            self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=10, stride=2, padding=4)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = x.squeeze(2)
        x = self.conv3(x)
        return x

class Region_Spatial_Encoder(nn.Module):
    def __init__(self,local_dict, n_layers, d_model, head, d_ff, dropout):
        super(Region_Spatial_Encoder, self).__init__()
        self.local_dict = local_dict
        self.region_att_dict = nn.ModuleDict()
        self.feed_forward_dict = nn.ModuleDict()

        for i in self.local_dict.keys():
            # spatial_region_dim = len(self.local_dict[i])
            region_att = MultiHeadAttention(head, d_model, dropout)
            feed_forward = FeedForward(d_model, d_ff)
            encoder_region = Encoder(n_layers, EncoderLayer(d_model, deepcopy(region_att), deepcopy(feed_forward)))

            self.region_att_dict[f'S_Region{i}_att'] = region_att
            self.feed_forward_dict[f'S_feed_forward{i}'] = feed_forward
            setattr(self, f'S_encoder_region{i}', encoder_region)

    def forward(self, x):
        regions = []
        encoded_regions = []

        # 根据给定的region进行channel维度的切分
        for i in self.local_dict.keys():
            region = x[:, self.local_dict[i], :]
            # regions.append(region)
            encoded_region = getattr(self, f'S_encoder_region{i}')(region, None)
            encoded_regions.append(encoded_region)
        # 对每个Region进行编码
        # for i in range(1, len(self.local_dict)):
        #     encoded_region = getattr(self, f'encoder_region{i}')(regions[i - 1], None)
        #     encoded_regions.append(encoded_region)

        return encoded_regions


class Region_Temporal_Encoder(nn.Module):
    def __init__(self,local_dict, n_layers, d_model, head, d_ff, dropout):
        super(Region_Temporal_Encoder, self).__init__()
        self.local_dict = local_dict
        self.region_att_dict = nn.ModuleDict()
        self.feed_forward_dict = nn.ModuleDict()

        for i in self.local_dict.keys():
            region_att = MultiHeadAttention(head, d_model, dropout)
            feed_forward = FeedForward(d_model, d_ff)
            encoder_region = Encoder(n_layers, EncoderLayer(d_model, deepcopy(region_att), deepcopy(feed_forward)))

            self.region_att_dict[f'T_Region{i}_att'] = region_att
            self.feed_forward_dict[f'T_feed_forward{i}'] = feed_forward
            setattr(self, f'T_encoder_region{i}', encoder_region)


    def forward(self, x):
        regions = []
        encoded_regions = []

        # 根据给定的region进行channel维度的切分
        for i in self.local_dict.keys():
            region = x[:, self.local_dict[i], :]
            encoded_region = getattr(self, f'T_encoder_region{i}')(region, None)
            encoded_regions.append(encoded_region)


        return encoded_regions


class Spatial_Squeeze_Excitation(nn.Module):
    def __init__(self, n_nodes):
        super(Spatial_Squeeze_Excitation, self).__init__()
        self.Linear1 = nn.Linear(n_nodes, n_nodes)  # 进行一个普通的全连接层变化，但不修改维度
        self.Linear2 = nn.Linear(n_nodes, n_nodes)  # 进行一个普通的全连接层变化，但不修改维度

    def forward(self, x):
        # global average pooling 针对一个region内部各
        x_f = x.mean(dim = -1)
        # 两层全链接，提取加权特征值
        x_f = self.Linear1(x_f)
        x_f = F.sigmoid(x_f)
        x_f = self.Linear2(x_f)
        x_f = F.sigmoid(x_f)
        x_f = x_f.unsqueeze(-1)
        # 逐通道加权,得到spatial的region内加权，不做任何的维度操作试试看
        weighted_sq = torch.mul(x, x_f)
        return weighted_sq

class Temporal_Squeeze_Excitation(nn.Module):
    def __init__(self, n_nodes):
        super(Temporal_Squeeze_Excitation, self).__init__()
        self.Linear1 = nn.Linear(n_nodes, n_nodes)  # 进行一个普通的全连接层变化，但不修改维度
        self.Linear2 = nn.Linear(n_nodes, n_nodes)  # 进行一个普通的全连接层变化，但不修改维度

    def forward(self, x):
        # global average pooling 针对一个region内部各
        x_f = x.mean(dim = 1)
        # 两层全链接，提取加权特征值
        x_f = self.Linear1(x_f)
        x_f = F.sigmoid(x_f)
        x_f = self.Linear2(x_f)
        x_f = F.sigmoid(x_f)
        x_f = x_f.unsqueeze(-2)
        # 逐通道加权,得到spatial的region内加权
        weighted_sq = torch.mul(x,x_f)
        return weighted_sq

class Region_Encoder(nn.Module):
    def __init__(self, spatial_local_dict, temporal_local_dict, n_layers, d_model_dict, head_dict, d_ff, dropout):
        super(Region_Encoder, self).__init__()

        # 词向量维度拆分
        s_d_model = d_model_dict['spatial']
        t_d_model = d_model_dict['temporal']

        # 头拆分
        s_head = head_dict['spatial']
        t_head = head_dict['temporal']

        # 划分通道的Region encoder部分
        self.Region_Spatial_Encoder = Region_Spatial_Encoder(spatial_local_dict, n_layers, s_d_model, s_head, d_ff, dropout)
        self.spatial_se_list = nn.ModuleList(
            [Spatial_Squeeze_Excitation(len(spatial_local_dict[i])) for i in spatial_local_dict.keys()])
        self.Linear1_Region_S = nn.Linear(len(spatial_local_dict), len(spatial_local_dict))
        self.Linear2_Region_S = nn.Linear(len(spatial_local_dict), len(spatial_local_dict))

        # 划分时间片段的Region encoder部分
        self.Region_Temporal_Encoder = Region_Temporal_Encoder(temporal_local_dict, n_layers, t_d_model, t_head, d_ff, dropout)
        self.temporal_se_list = nn.ModuleList(
            [Spatial_Squeeze_Excitation(len(temporal_local_dict[i]))for i in temporal_local_dict.keys()])
        self.Linear1_Region_T = nn.Linear(len(temporal_local_dict), len(temporal_local_dict))
        self.Linear2_Region_T = nn.Linear(len(temporal_local_dict), len(temporal_local_dict))

    def forward(self, x):

        # Region内部的时空Transformer提取特征
        x_t = x.permute(0,2,1)
        s_regional_encoded = self.Region_Spatial_Encoder(x)
        t_regional_encoded = self.Region_Temporal_Encoder(x_t)

        # 逐个空间channel和时间point的加权----第一层L2G
        s_ch_L2G_ls = []
        for s_idx in range(len(s_regional_encoded)):
            se = self.spatial_se_list[s_idx](s_regional_encoded[s_idx])
            represent = se.mean(1).mean(1).unsqueeze(1)
            s_ch_L2G_ls.append(represent)
        region_se_x = torch.cat(s_ch_L2G_ls, 1)
        region_se_x = self.Linear1_Region_S(region_se_x)
        region_se_x = F.softmax(region_se_x, dim=1)
        region_se_x = self.Linear2_Region_S(region_se_x)
        region_se_x = F.softmax(region_se_x, dim=1)

        # 每个区域划区块空间channel和时间point的加权----第二层
        spatial_encoded_ret = []
        for r_i in range(len(s_regional_encoded)):
            spatial_encoded_ret.append(region_se_x[:, r_i].unsqueeze(-1).unsqueeze(-1) * s_regional_encoded[r_i])

        t_tm_L2G_ls = []
        for t_idx in range(len(t_regional_encoded)):
            te = self.temporal_se_list[t_idx](t_regional_encoded[t_idx])
            represent = te.mean(1).mean(1).unsqueeze(1)
            t_tm_L2G_ls.append(represent)
        region_te_x = torch.cat(t_tm_L2G_ls, 1)
        region_te_x = self.Linear1_Region_T(region_te_x)
        region_te_x = F.softmax(region_te_x, dim=1)
        region_te_x = self.Linear2_Region_T(region_te_x)
        region_te_x = F.softmax(region_te_x, dim=1)

        # 每个区域划区块空间channel和时间point的加权----第二层
        temporal_encoded_ret = []
        for r_i in range(len(t_regional_encoded)):
            temporal_encoded_ret.append(region_te_x[:, r_i].unsqueeze(-1).unsqueeze(-1) * t_regional_encoded[r_i])

        # 合并操作
        Spatial_ret =  torch.cat(spatial_encoded_ret, dim=1)
        Temporal_ret = torch.cat(temporal_encoded_ret, dim=1).permute(0,2,1)
        ret = (Temporal_ret + Spatial_ret) / 2
        return ret


##----------------------------- 模型整体骨干-----------------------------##
class STL2G(nn.Module):
    def  __init__(self, d_model_dict, head_dict ,d_ff, n_layers, spatial_local_dict, temporal_local_dict,
                  dropout, clf_class=4, domain_class=8):
        super(STL2G, self).__init__()

        ## Temporal_emb
        self.temporal_embedding = nn.Sequential()
        self.temporal_embedding.add_module('Temporal_embedding', Temporal_Embedding(30, d_model_dict['temporal'], dropout))
        # self.temporal_embedding.add_module('Temporal_embedding', shallow_conv())

        ## Spatial-Temporal Local to Global ##
        self.spatial_L_to_G = nn.Sequential()
        self.spatial_L_to_G.add_module('Spatial_L_G', Region_Encoder(spatial_local_dict, temporal_local_dict, n_layers=n_layers,
                                                                     d_model_dict= d_model_dict , head_dict = head_dict, d_ff=d_ff, dropout=dropout))

        # Class classifier layer
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc2', nn.Linear(d_model_dict['spatial']*d_model_dict['temporal'], clf_class))
        # self.class_classifier.add_module('c_fc1', nn.Linear(d_model_dict['spatial']*d_model_dict['temporal'], 8))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(8))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout(dropout))
        # self.class_classifier.add_module('c_fc2', nn.Linear(8, clf_class))
        #
        # Domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(d_model_dict['spatial']*d_model_dict['temporal'], domain_class))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(8))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_drop1', nn.Dropout(dropout))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(8, domain_class))

    def forward(self, x, alpha):
        x = self.temporal_embedding(x)   # 2, ch, 62
        x = self.spatial_L_to_G(x)
        # feature = self.temporal_L_to_G(x)
        ### channel 维度的加权 to-Global
        # [2,22,24]
        feature = x.reshape(-1, x.shape[1] * x.shape[2])
        # feature = torch.mean(x, 1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output



if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 30, 400))
    s_head = 2
    d_head = 2
    s_d_model = 62
    t_d_model = 22
    d_model_dict = {'spatial':30, 'temporal':30}
    head_dict = {'spatial':5, 'temporal':5}
    d_ff = 128
    ff_hide = 1024
    n_layers = 3
    dropout = 0.1
    n_heads = 4
    spatial_local_dict = {'1': [0, 2, 3, 4], '2': [6, 1, 7, 13], '3': [i for i in range(18,30)], '4': [5, 11, 17, 12], '5': [8, 9, 10, 14, 15, 16]}
    temp_local_dict = {
            '1':[i for i in range(10)],
            '2':[i for i in range(10,15)],
            '3':[i for i in range(15,30)],
        }
    # model = Temporal_Embedding(30, 30, 0.3)
    model = STL2G(d_model_dict, head_dict, d_ff, n_layers, spatial_local_dict, temp_local_dict, dropout, clf_class=2,
                  domain_class=8)
    res,res2 = model(inp, 0.01)

    print(model)

