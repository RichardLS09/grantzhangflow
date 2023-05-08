import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


def default_collate(inp,device,dtype):
  return Tensor(
    inp,
    device=device,
    dtype=dtype,
    requires_grad=False
  )

def collate_ndarray(inp,device,dtype):
  if isinstance(inp,np.ndarray):
    return inp
  assert hasattr(inp,"numpy")
  return inp.numpy()


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    import gzip
    import numpy as np
    
    def _bytes_to_int(b):
        return int.from_bytes(b,byteorder="big")
    
    images = []
    IMAGE_ROW = 28
    IMAGE_COL = 28
    with gzip.open(image_filesname,"rb") as f:
        magic_number = _bytes_to_int(f.read(4))
        num_images = _bytes_to_int(f.read(4))
        num_rows = _bytes_to_int(f.read(4))
        num_cols = _bytes_to_int(f.read(4))
        # assert magic_number == 2051
        # assert num_images == 60000
        # assert num_rows == IMAGE_ROW
        # assert num_cols == IMAGE_COL
        for i in f.read():
            images.append(int(i))
    i_max = max(images)
    i_min = min(images)
    out = [[]]
    for i in images:
        if len(out[-1]) == IMAGE_ROW * IMAGE_COL:
            out.append([])
        out[-1].append((i-i_min)/(i_max-i_min))
    out = np.array(out,dtype="float32")
    
    labels = []
    with gzip.open(label_filename,"rb") as f:
        magic_number = _bytes_to_int(f.read(4))
        num_labels = _bytes_to_int(f.read(4))
        
        for i in f.read():
            labels.append(int(i))
    # print(max(labels),min(labels))
    labels = np.array(labels,dtype="uint8")
    assert len(out) == len(labels)
    return out,labels

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        # print(img.shape, flip_img)
        if flip_img:
          assert len(img.shape) in (1, 2,3)
          if len(img.shape) == 3:
            return img[:,::-1,:]
          elif len(img.shape) == 2:
            return img[:,::-1]
          else:
            new_shape = int(np.sqrt(img.shape[0]).astype("int"))
            return img.reshape((new_shape,new_shape))[:,::-1].reshape(img.shape)
        else:
          return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        img_shape = list(img.shape)
        assert len(img_shape) in (1, 2,3)
        if len(img_shape) == 2:
          pad_shape = [(self.padding,self.padding), (self.padding,self.padding)]
        elif len(img_shape) == 3:
          pad_shape = [(self.padding,self.padding), (self.padding,self.padding),(0,0)]
        else:
          pad_shape = [(self.padding,self.padding), (self.padding,self.padding)]
          new_shape = int(np.sqrt(img.shape[0]).astype("int"))
          img = img.reshape((new_shape,new_shape))

        # print(img.shape,pad_shape)
        pad_img = np.pad(img,pad_shape,)
        shift_x += self.padding
        shift_y += self.padding
        if len(img_shape) == 2:
          return pad_img[shift_x:shift_x+img_shape[0], shift_y:shift_y+img_shape[1]]
        if len(img_shape) == 3:
          return pad_img[shift_x:shift_x+img_shape[0], shift_y:shift_y+img_shape[1],:]
        return pad_img[shift_x:shift_x+new_shape, shift_y:shift_y+new_shape].reshape(img_shape)
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        collate_fn = None,
        device = None,
        dtype = "float32"
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )
        self.dtype = dtype
        self.device = device
        self.collate_fn = collate_fn or default_collate
        assert self.collate_fn in (default_collate,collate_ndarray)

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self._data_cnt = 0
        if self.shuffle:
          # if not hasattr(self,"ordering"):
            tmp = np.arange(len(self.dataset))
            np.random.shuffle(tmp)
            self.ordering = np.array_split(tmp, 
                                            range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self._data_cnt < len(self.ordering):
          batch_inidex = self.ordering[self._data_cnt]
          self._data_cnt += 1
          out = self.dataset[batch_inidex]
          if isinstance(out,(list,tuple)):
            return tuple(
              (self.collate_fn(nd.array(i),device=self.device,
              dtype=self.dtype
            ) for i in out))
          else:
            return self.collate_fn(nd.array(out), device=self.device,
              dtype=self.dtype
            )
        raise StopIteration
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        image,label = parse_mnist(image_filename,label_filename)
        image = image.reshape((image.shape[0],28,28,1))
        self.data = NDArrayDataset(image,label)
        super().__init__(transforms)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        i,l = self.data[index]
        return self.apply_transforms(i),l
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.data)
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.train = train
        self.p = p

        def load_pkl(filename):
          with open(filename, "rb") as f:
              batch_data = pickle.load(f, encoding="latin1")
          return batch_data

        meta_file = os.path.join(base_folder,"batches.meta")
        # num_cases_per_batch, label_names, num_vis
        meth_data = load_pkl(meta_file)

        data = []
        label = []
        if self.train:
          for i in range(1,6):
            batch_file = os.path.join(base_folder,f"data_batch_{i}")
            # batch_label, labels, data, filenames
            batch_data = load_pkl(batch_file)
            data.extend(batch_data["data"])
            label.extend(batch_data["labels"])
        else:
          test_file = os.path.join(base_folder,f"test_batch")
          test_data = load_pkl(test_file)
          data.extend(test_data["data"])
          label.extend(test_data["labels"])

        data = np.array(data,dtype="float32")
        label = np.array(label,dtype="uint8")
        data = data / 255
        data = data.reshape(len(data),3,32,32)
        self.X = data
        self.Y = label

        self.data = NDArrayDataset(self.X,self.Y)

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        i,l = self.data[index]
        i,l = self.apply_transforms(i),l
        # i,l = nd.array(i), nd.array(l)
        # return Tensor.make_const(i), Tensor.make_const(l)
        return i,l
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.data)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word in self.word2idx:
          return self.word2idx[word]
        idx = len(self.idx2word)
        self.idx2word.append(word)
        self.word2idx[word] = idx
        return idx
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        all_ids = []
        with open(path,"r") as f:
          for line_index,line in enumerate(f,1):
            if max_lines is not None and line_index>max_lines:
              break
            t_ids = []
            line = line[1:-1] + "<eos>"
            for word in line.split(" "):
              n_id = self.dictionary.add_word(word)
              t_ids.append(n_id)
              all_ids.append(n_id)
            ids.append(t_ids)
        return all_ids

        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    length = len(data)
    n_batch = length // batch_size
    nums = n_batch * batch_size
    return np.array(data[:nums],dtype=dtype).reshape([batch_size,n_batch]).transpose()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    seq_length = min(bptt, len(batches)-1-i)
    data = batches[i:i+seq_length]
    target = batches[i+1:i+seq_length+1].reshape([np.prod(data.shape),])
    return (Tensor(data, device=device,dtype=dtype,requires_grad=False),
        Tensor(target, device=device,dtype=dtype,requires_grad=False)
    )

    ### END YOUR SOLUTION