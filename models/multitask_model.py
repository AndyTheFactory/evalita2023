
from torch import nn
from transformers import BertModel, BertTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class MultiTaskModel(nn.Module):
    def __init__(self, n_classes, dropout=0.3, pre_trained_model:str = 'bert-base-uncased') -> None:
        super(MultiTaskModel,self).__init__()

        self.bert = BertModel.from_pretrained(pre_trained_model)
        self.dropout = nn.Dropout(p=dropout)

        self.hidden_size = self.bert.config.hidden_size

        self.output_heads = nn.ModuleDict()
        self.output_heads['semeval'] = nn.Linear(self.hidden_size, n_classes)
        self.output_heads['mlm'] =  BertOnlyMLMHead(self.bert.config)
        
    def forward(self, input_ids, task='a'):
        
        out = self.bert(input_ids=input_ids, return_dict=True)
        
        if task != 'mlm':
            out = self.dropout(out.pooler_output)
            task = 'semeval'
        else:
            out = out.last_hidden_state


        out = self.output_heads[task](out)

        return out

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def get_tokenizer(pre_trained_model:str = 'bert-base-uncased') -> BertTokenizer:
        return BertTokenizer.from_pretrained(pre_trained_model)
