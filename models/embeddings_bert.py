import torch
from torch import nn
from transformers import BertModel, BertTokenizer


class EmbeddingsBERT(nn.Module):
  def __init__(self, n_classes, dropout=0.3, pre_trained_model:str = 'bert-base-uncased', embedding_size=1536) -> None:
      super(EmbeddingsBERT,self).__init__()

      self.bert = BertModel.from_pretrained(pre_trained_model)
      self.dropout = nn.Dropout(p=dropout)

      self.output = nn.Linear(self.bert.config.hidden_size+embedding_size, n_classes)

  def forward(self, input_ids, embedding, **kwargs):

    _, pooled_o = self.bert(input_ids=input_ids, return_dict=False)
    out = torch.cat([pooled_o, embedding], dim=1)
    out = self.dropout(out)

    return self.output(out)

  @property
  def device(self):

    return next(self.parameters()).device

  @staticmethod
  def get_tokenizer(pre_trained_model:str = 'bert-base-uncased') -> BertTokenizer:
    return BertTokenizer.from_pretrained(pre_trained_model)

