from pathlib import Path

class Config:
  bert_model = 'bert-base-uncased'
  epochs = 5
  data_folder = Path(__file__).parent.parent.parent / 'data'
