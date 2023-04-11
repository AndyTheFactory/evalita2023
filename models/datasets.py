import random
from typing import Union
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import pandas as pd
from transformers import AutoTokenizer
from config import Config

class EvalitaDataset(Dataset):
  def __init__(self, df: pd.DataFrame,*, task:str, max_len:int=64, df_emb:pd.DataFrame=None):
    assert task in ['a','b', None], 'Choose Tasks a / b  or None for test set'
    self.df = df
    self.df_emb = df_emb
    self.task = task
    if self.task=='a':
      self.label_f = 'conspiratorial'
    elif self.task=='b':
      self.label_f = 'conspiracy'
    else:
      self.label_f = None
   
    self.tokenizer = AutoTokenizer.from_pretrained(Config.bert_model)
    self.max_len = max_len

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    rec = self.df.iloc[idx]

    label = rec[self.label_f] if self.label_f else 0

    encoded = self.tokenizer.encode(rec.comment_text, 
                                    max_length=self.max_len,
                                    padding='max_length', truncation=True)

    embedding = None
    if self.df_emb is not None:
      x = self.df_emb.loc[self.df_emb.comment_text==rec.comment_text]['embedding-ada-002']
      # x is a string containing a list of values
      if len(x)>0:
        embedding = torch.tensor([float(i) for i in x.values[0][1:-1].split(',')])


    return {
        'record_id': rec.Id,
        'input_ids': torch.tensor(encoded),
        'embedding': embedding,
        'label': label,
        'task': self.task,

    }


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

        
def get_dataloader(csv_file:Union[str,pd.DataFrame], task:str, batch_size:int, max_len:int=64, shuffle:bool=True, **kwargs):
  if isinstance(csv_file, str):
    df = pd.read_csv(csv_file)
  else:
    df = csv_file

  dataset = EvalitaDataset(df, task=task, max_len=max_len, **kwargs)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return dataloader

def get_dataloader_eval(task:str, tokenizer, batch_size:int, max_len:int=64):
  task_file = Config.data_folder / f'subtask{task.upper()}_test.csv'
  return get_dataloader(task_file, None, tokenizer, batch_size, max_len, False)

