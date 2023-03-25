from torch import nn
from transformers import BertModel, BertTokenizer, BertLayer


class TestBERT(nn.Module):
  def __init__(self,  n_classes, dropout=0.3, pre_trained_model:str = 'bert-base-uncased', add_layers:int=1) -> None:
      super(TestBERT,self).__init__()

      self.bert = BertModel.from_pretrained(pre_trained_model)
      for param in self.bert.parameters():
        param.requires_grad = False
      for _ in range(add_layers):
        self.bert.encoder.layer.append(BertLayer(self.bert.config))
      self.bert.config.num_hidden_layers += add_layers

      self.dropout = nn.Dropout(p=dropout)
      self.output = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids):

    _, pooled_o = self.bert(input_ids=input_ids, return_dict=False)
    out = self.dropout(pooled_o)

    return self.output(out)

  @property
  def device(self):

    return next(self.parameters()).device

  @staticmethod
  def get_tokenizer(pre_trained_model:str = 'bert-base-uncased') -> BertTokenizer:
    return BertTokenizer.from_pretrained(pre_trained_model)

