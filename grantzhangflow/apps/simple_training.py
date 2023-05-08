import sys
sys.path.append('../python')
import inspect
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time
import numpy as np

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
      model.train()
    else:
      model.eval()
    
    if inspect.isclass(loss_fn):
      loss_fn = loss_fn()

    wrong = 0
    loss_all = 0
    loss_list = []
    for index,(d,label) in enumerate(dataloader):
      if opt is not None:
        opt.reset_grad()
      if not isinstance(d,ndl.Tensor):
        d = ndl.autograd.Tensor(d,device=device,requires_grad=False)
      if not isinstance(label,ndl.Tensor):
        label = ndl.autograd.Tensor(label,device=device,requires_grad=False)
      y = model(d)
      loss = loss_fn(y,label)
      if opt is not None:
        loss.backward()

      this_loss = float(loss.detach().numpy())
      loss_list.append(this_loss)
      loss_all += this_loss * d.shape[0]
      y_numpy = y.detach().numpy()
      label_numpy = label.detach().numpy()
      wrong += int(np.sum(
        np.argmax(y_numpy,axis=-1) == label_numpy
      ))

      if opt is not None:
        opt.step()
    error_rate = wrong / len(dataloader.dataset)
    one_loss = loss_all / len(dataloader.dataset)
    two_loss = float(np.mean(loss_list))
    return error_rate, two_loss

    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    for i in range(n_epochs):
      # print("run epoch",i)
      avg_acc, avg_loss = epoch_general_cifar10(dataloader,model,loss_fn=loss_fn,
        opt=opt)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_cifar10(dataloader,model,loss_fn=loss_fn,opt=None)
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
      model.train()
    else:
      model.eval()
    
    if inspect.isclass(loss_fn):
      loss_fn = loss_fn()

    wrong = 0
    loss_all = 0
    loss_list = []

    nbatch, batch_size = data.shape

    data_length = len(data)

    tmp = np.arange(batch_size)
    np.random.shuffle(tmp)
    # data = data[:, tmp]
    
    data = [
      ndl.data.get_batch(data,i,seq_len,device=device, dtype=dtype)
      # for i in range(0,data.shape[0]-seq_len,)
      for i in range(0,nbatch-1,seq_len)
    ]
    # data = data[:2]
    


    for index,(d,label) in enumerate(data):
      if opt is not None:
        opt.reset_grad()
      # if not isinstance(d,ndl.Tensor):
      #   d = ndl.autograd.Tensor(d,device=device,requires_grad=False)
      # if not isinstance(label,ndl.Tensor):
      #   label = ndl.autograd.Tensor(label,device=device,requires_grad=False)
      y,_ = model(d)
      loss = loss_fn(y,label)
      if opt is not None:
        loss.backward()
        if clip is None:
          opt.clip_grad_norm()
        else:
          opt.clip_grad_norm(clip)

      this_loss = float(loss.detach().numpy())
      loss_list.append(this_loss)
      loss_all += this_loss * d.shape[0]
      y_numpy = y.detach().numpy()
      label_numpy = label.detach().numpy()
      wrong += int(np.sum(
        np.argmax(y_numpy,axis=-1) == label_numpy
      ))
      if opt is not None:
        opt.step()
    error_rate = wrong / (batch_size * len(data))
    one_loss = loss_all / (batch_size * len(data))
    two_loss = float(np.mean(loss_list))
    return error_rate, two_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    for i in range(n_epochs):
      # print("run epoch",i)
      avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=opt,
        clip=clip, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=None,
        clip=None, device=device, dtype=dtype)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
