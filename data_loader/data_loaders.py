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

