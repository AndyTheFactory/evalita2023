from datetime import datetime
import logging
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
import pandas as pd


DATA_FOLDER = Path(__file__).parent.parent.parent / 'data'

LABEL_MAPS = {
    'a': {
            'not sexist':0, 
            'sexist':1
        },
    'b': {
            'none':-1, 
            '1. threats, plans to harm and incitement':0,
            '2. derogation':1, 
            '3. animosity':2,
            '4. prejudiced discussions':3,
          },
    'c': {
          '1.1 threats of harm': 0,
          '1.2 incitement and encouragement of harm': 1,
          '2.1 descriptive attacks': 2,
          '2.2 aggressive and emotive attacks': 3,
          '2.3 dehumanising attacks & overt sexual objectification': 4,
          '3.1 casual use of gendered slurs, profanities, and insults': 5,
          '3.2 immutable gender differences and gender stereotypes': 6,
          '3.3 backhanded gendered compliments': 7,
          '3.4 condescending explanations or unwelcome advice': 8,
          '4.1 supporting mistreatment of individual women': 9,
          '4.2 supporting systemic discrimination against women as a group': 10,
          'none': -1, 
        }
}

def label_decode(task, label):
  d = LABEL_MAPS[task]
  d = dict(zip(d.values(), d.keys()))
  return d[label]

def make_submission(task, predictions, output_folder):
  output_file = output_folder / datetime.now().strftime(f'submission_%Y%m%d-%H%M%S-task-{task}.csv') 
  df = pd.read_csv(DATA_FOLDER / f'dev_task_{task}_entries.csv')
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