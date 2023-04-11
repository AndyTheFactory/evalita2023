from datetime import datetime
import logging
from zipfile import ZipFile, ZIP_DEFLATED
from config import Config
import pandas as pd

def task_to_classes(task:str)->int:
  if task=='a':
    return 2
  elif task=='b':
    return 3
  else:
    return 0


def make_submission(task, predictions, output_folder):
  output_file = output_folder / datetime.now().strftime(f'submission_%Y%m%d-%H%M%S-task-{task}.csv') 
  df = pd.read_csv(Config.data_folder / f'dev_task_{task}_entries.csv')
  df['label_pred']  = list(map(lambda x: label_decode(task, x), predictions))
  df = df[df['label_pred']!='none']
  df[['rewire_id', 'label_pred']].to_csv(output_file, index=None)

  with ZipFile(output_file.with_suffix('.zip'), 'w', compression=ZIP_DEFLATED, compresslevel=9) as zip:
    zip.write(output_file, arcname=output_file.name)
    output_file.unlink()

  logger = logging.getLogger()
  logger.info(f"File for Task {task} written: {output_file}")

def set_seed(seed:int):
  import random
  import numpy as np
  import torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False