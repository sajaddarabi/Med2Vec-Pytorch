from torchvision import datasets, transforms
from .med2vec_dataset import Med2VecDataset
from .med2vec_dataset import collate_fn as med2vec_collate
from .med2vec_dataset_transformer import TransformerMedDS
from .med2vec_dataset_transformer import collate_fn as structmed_collate
from .mortality_dataset import MortalityDataset
from .mortality_dataset import collate_fn as mortality_collate


from .text2code_dataset import Text2CodeDataset
from .text2code_dataset import collate_fn as text2code_collate

from .text_dataset import TextDataset
from .text_dataset import collate_fn as text_collate

from .los_readmission_dataset import LosReadmissionDataset
from .los_readmission_dataset import collate_fn as losred_collate

from base import BaseDataLoader

import os
import pickle
class Med2VecDataLoader(BaseDataLoader):
    """
    Med2Vec Dataloader
    """
    def __init__(self, data_dir, num_codes, batch_size, shuffle, validation_split, num_workers, med=True, diag=True, proc=True, file_name=None, training=True, dict_format=False):
       self.data_dir = data_dir
       self.num_codes = num_codes
       path_file = os.path.expanduser(data_dir)

       if file_name != None:
           path_file = os.path.join(path_file, file_name)
       else:
           path_file = os.path.join(path_file, 'med2vec.seqs')
       self.data_dir = path_file
       self.train = training
       self.dataset = Med2VecDataset(self.data_dir, self.num_codes, dict_format, self.train, med=med, diag=diag, proc=proc)
       super(Med2VecDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=med2vec_collate)

class StructMedDataLoader(BaseDataLoader):
    """
    Med2Vec Dataloader
    """
    def __init__(self, data_dir, num_codes, batch_size, shuffle, validation_split, num_workers, vocab_fname='',training=True, file_name=None):
        self.root = os.path.expanduser(data_dir)
        self.data_path = os.path.join(self.root, 'data.pkl')
        if (vocab_fname == ''):
            self.vocab_path = os.path.join(self.root, 'vocab.pkl')
        else:
            self.vocab_path = os.path.join(self.root, vocab_fname,)
        self.num_codes = num_codes
        self.train = training
        self.batch_size = batch_size
        self.dataset = TransformerMedDS(self.data_path, self.vocab_path, self.num_codes, self.batch_size, self.train)
        super(StructMedDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
               collate_fn=structmed_collate)

class MortalityDataLoader(BaseDataLoader):
    """
    Mortality prediction task
    """
    def __init__(self, data_dir, vocab_path, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_path = data_dir
        self.vocab_path = vocab_path
        data_path_file = os.path.expanduser(self.data_path)
        vocab_path_file = os.path.expanduser(vocab_path)
        self.batch_size = batch_size
        self.train = training
        self.dataset = MortalityDataset(self.data_path, self.vocab_path, self.batch_size, self.train)
        super(MortalityDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                collate_fn=mortality_collate)

class TextDataLoader(BaseDataLoader):
    """
    Mortality prediction task
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.root = os.path.expanduser(data_dir)
        self.data_path = os.path.join(self.root, 'data.pkl')
        self.vocab_path = os.path.join(self.root, 'vocab.pkl')
        self.batch_size = batch_size
        self.train = training
        self.dataset = TextDataset(self.data_path, self.vocab_path, self.batch_size, self.train)
        self.vocab = pickle.load(open(self.vocab_path, 'rb'))

        super(TextDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                collate_fn=text_collate)

class LosReadmissionDataLoader(BaseDataLoader):
    """
    Mortality prediction task
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, y_label='los', training=True, vocab_fname=''):
        self.root = os.path.expanduser(data_dir)
        self.data_path = os.path.join(self.root, 'data.pkl')
        if (vocab_fname == ''):
            self.vocab_path = os.path.join(self.root, 'vocab.pkl')
        else:
            self.vocab_path = os.path.join(self.root, vocab_fname)
        self.batch_size = batch_size
        self.train = training

        self.dataset = LosReadmissionDataset(self.data_path, self.vocab_path, self.batch_size, y_label, self.train)

        super(LosReadmissionDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                collate_fn=losred_collate)
