import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass, field
from sklearn.metrics import f1_score, classification_report

@dataclass
class TrainInfo:
  train_acc: list = field(default_factory=list)
  train_f1: list = field(default_factory=list) 
  train_loss: list = field(default_factory=list)
  best_f1: float = 0
  best_epoch: int = 0
  best_acc: float = 0
  best_loss: float = float("inf")

  def __repr__(self) -> str:
    return f"""
    Nr epochs: {len(self.train_acc)}
    Best epoch: {self.best_epoch}

    Best acc: {self.best_acc}
    Best f1: {self.best_f1}
    Best loss: {self.best_loss}
    """

  def add_train_info(self, train_acc, train_f1, train_loss):
    self.train_acc.append(train_acc)
    self.train_f1.append(train_f1)
    self.train_loss.append(train_loss)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = 0
        self.best_loss = np.inf

    def early_stop(self, metric):
        if metric > self.best_metric:
            self.best_metric = metric
            self.counter = 0
        elif metric < (self.best_metric - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def early_stop_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        elif loss > (self.best_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def data_to_device(d, device):
  if isinstance(d['input_ids'],list):
      d['input_ids'] = torch.stack(d['input_ids'],1)
  for k in d:
    if isinstance(d[k],torch.Tensor):
      d[k] = d[k].to(device)        
  return d

def train_epoch(model, data_loader, loss_fn, optimizer,  scheduler, epoch=1):

  model = model.train()
  device = model.device
  losses = []
  predict_out = []
  all_label_ids = []

  correct_predictions = 0

  for d in tqdm(data_loader, desc=f'Training Epoch {epoch} '):

    d = data_to_device(d, device)
  
    targets = d["label"]
    outputs = model(**d)

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()

    predict_out.extend(preds.tolist())
    all_label_ids.extend(targets.tolist())

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  f1_metrics=f1_score(np.array(all_label_ids).reshape(-1),
      np.array(predict_out).reshape(-1), average='weighted')
  report = classification_report(np.array(all_label_ids).reshape(-1),
      np.array(predict_out).reshape(-1),digits=4)
  print("Report:\n"+report)
  n_examples = len(all_label_ids)
  return correct_predictions.double() / n_examples, f1_metrics, report, np.mean(losses)


def eval_model(model, data_loader, loss_fn):

  model = model.eval()
  device = model.device
  losses = []
  predict_out = []
  all_label_ids = []

  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader, desc='Evaluation Process'):
      
      d = data_to_device(d, device)
      targets = d["label"]
      outputs = model(**d)

      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
      predict_out.extend(preds.tolist())
      all_label_ids.extend(targets.tolist())

  n_examples = len(all_label_ids)

  f1_metrics=f1_score(np.array(all_label_ids).reshape(-1),
      np.array(predict_out).reshape(-1), average='weighted')
  report = classification_report(np.array(all_label_ids).reshape(-1),
      np.array(predict_out).reshape(-1),digits=4)
  print("Report:\n"+report)

  return correct_predictions.double() / n_examples, f1_metrics, report,  np.mean(losses)  

def predict(model,data_loader):

  model = model.eval()
  device = model.device

  predict_out = []

  with torch.no_grad():
    for d in tqdm(data_loader, desc='Predict data'):

      d = data_to_device(d, device)
      outputs = model(**d)
      
      _, preds = torch.max(outputs, dim=1)

      predict_out.extend(preds.tolist())

  return predict_out
