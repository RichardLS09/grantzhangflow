import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

import torch

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        def get_conv_bn(i_c,o_c,k,s):
          return nn.Sequential(
            nn.Conv(i_c,o_c,k,s,device=device,dtype=dtype),
            nn.BatchNorm2d(dim=o_c,device=device,dtype=dtype),
            nn.ReLU()
          )
        shape_list = [
          (3,16,7,4),
          (16,32,3,2),
          (32,32,3,1),
          (32,32,3,1),

          (32,64,3,2),
          (64,128,3,2),
          (128,128,3,1),
          (128,128,3,1),
        ]
        
        self.c1 = get_conv_bn(*shape_list[0])
        self.c2 = get_conv_bn(*shape_list[1])
        self.c3 = get_conv_bn(*shape_list[2])
        self.c4 = get_conv_bn(*shape_list[3])

        self.c5 = get_conv_bn(*shape_list[4])
        self.c6 = get_conv_bn(*shape_list[5])
        self.c7 = get_conv_bn(*shape_list[6])
        self.c8 = get_conv_bn(*shape_list[7])

        self.flat = nn.Flatten()

        self.l1 = nn.Linear(128,128,device=device,dtype=dtype)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128,10,device=device,dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION

        def one_run(inp,c1,c2,c3,c4):
          o0 = inp
          o1 = c1(o0)
          o2 = c2(o1)
          o3 = c3(o2)
          o4 = c4(o3)
          out = o2 + o4
          return out
        o = one_run(x,self.c1,self.c2,self.c3,self.c4)
        o = one_run(o,self.c5,self.c6,self.c7,self.c8)

        o = self.flat(o) # val
        o = self.l1(o) # val
        o = self.relu(o) # val
        o = self.l2(o) # val

        return o

        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_layer = nn.Embedding(
          output_size,embedding_size,device,dtype
        )
        seq_cls = nn.LSTM if seq_model == "lstm" else nn.RNN
        self.seq_layer = seq_cls(
          embedding_size,hidden_size,
          num_layers=num_layers,device=device,
          dtype=dtype
        )
        self.fc = nn.Linear(
          hidden_size,output_size,
          device=device,
          dtype=dtype
        )
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        embed_x = self.embed_layer(x)
        seq_x,res_h = self.seq_layer(embed_x,h)
        f_x = ndl.ops.reshape(seq_x,[seq_len * bs, self.hidden_size])
        res_x = self.fc(f_x)
        return res_x,res_h 
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)