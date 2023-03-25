import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import pandas as pd
from utils import DATA_FOLDER, LABEL_MAPS

class SexistDataset(Dataset):
  def __init__(self, df: pd.DataFrame, task:str, tokenizer, max_len:int=64):
    assert task in ['a','b','c', None], 'Choose Tasks a / b / c'
    self.df = df if task == 'a' or task is None else df[df['label_sexist'] == 'sexist']
    self.task = task
    self.label_f = getattr(self, 'label_'+self.task) if self.task else None
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    rec = self.df.iloc[idx]

    label = self.label_f(rec) if self.label_f else 0

    encoded = self.tokenizer.encode(rec.text, 
                                    max_length=self.max_len,
                                    padding='max_length', truncation=True)

    return {
        'record_id': rec.rewire_id,
        'input_ids': torch.tensor(encoded),
        'label': label,
        'task': self.task,

    }

  def label_a(self, rec)->int:
    label = rec['label_sexist']
    return LABEL_MAPS['a'][label]
  
  def label_b(self, rec)->int:
    label = rec['label_category']
    return LABEL_MAPS['b'][label]

  def label_c(self, rec)->int:
    label = rec['label_vector']
    return  LABEL_MAPS['c'][label]

class MLMDataset(Dataset):
  def __init__(self, df: pd.DataFrame, tokenizer, max_len:int=64):
    self.df = df

    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    rec = self.df.iloc[idx]


    encoded = self.tokenizer.encode(rec.text, 
                                    max_length=self.max_len,
                                    padding='max_length', truncation=True)

    encoded = torch.tensor(encoded)
    rand = torch.rand(torch.count_nonzero(encoded) - 2)    
    rand = torch.nn.functional.pad(rand,(0,encoded.shape[0]-rand.shape[0]),value=1)                  

    mask = (rand < 0.15) * (encoded != self.tokenizer.cls_token_id) * (
        encoded != self.tokenizer.sep_token_id) * (encoded != self.tokenizer.pad_token_id)

    return {
        'record_id': idx,
        'input_ids': encoded.masked_fill(mask, self.tokenizer.mask_token_id),         
        'label': encoded.masked_fill(~mask, 0),  
        'task': 'mlm'

    }

  def label_a(self, rec)->int:
    label = rec['label_sexist']
    return LABEL_MAPS['a'][label]
  

def chunk(indices, size):
    return torch.split(torch.tensor(indices), size)

class MyBatchSampler(Sampler):
    def __init__(self, a_indices, b_indices, batch_size): 
        self.a_indices = a_indices
        self.b_indices = b_indices
        self.batch_size = batch_size

    def __len__(self):
        return len(self.a_indices) + len(self.b_indices)
    
    def __iter__(self):
        random.shuffle(self.a_indices)
        random.shuffle(self.b_indices)
        a_batches  = chunk(self.a_indices, self.batch_size)
        b_batches = chunk(self.b_indices, self.batch_size)
        all_batches = list(a_batches + b_batches)
        all_batches = [batch.tolist() for batch in all_batches]
        random.shuffle(all_batches)
        return iter(all_batches)

        
def get_dataloader(csv_file:str, task:str, tokenizer, batch_size:int, max_len:int=64, shuffle:bool=True):
  df = pd.read_csv(csv_file)
  dataset = SexistDataset(df, task, tokenizer, max_len)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return dataloader

def get_dataloader_eval(task:str, tokenizer, batch_size:int, max_len:int=64):
  task_file = DATA_FOLDER / f'dev_task_{task}_entries.csv'
  return get_dataloader(task_file, None, tokenizer, batch_size, max_len, False)


def get_dataloader_multi(csv_file1:str, csv_file2:str, tokenizer, batch_size:int, max_len:int=64):
  df1 = pd.read_csv(csv_file1)
  dataset1 = SexistDataset(df1[:30], 'a', tokenizer, max_len)

  df2 = pd.read_csv(csv_file2)
  dataset2 = MLMDataset(df2[:30], tokenizer, max_len)

  joined = ConcatDataset([dataset1, dataset2])

  a_len = dataset1.__len__()
  ab_len = a_len + dataset2.__len__()
  a_indices = list(range(a_len))
  b_indices = list(range(a_len, ab_len))

  batch_sampler = MyBatchSampler(a_indices, b_indices, batch_size)

  dataloader = DataLoader(joined, batch_sampler=batch_sampler)
       
  return dataloader
