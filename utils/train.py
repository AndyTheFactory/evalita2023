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


def train_epoch(model, data_loader, loss_fn, optimizer,  scheduler, epoch=1):

  model = model.train()
  device = model.device
  losses = []
  predict_out = []
  all_label_ids = []

  correct_predictions = 0

  for d in tqdm(data_loader, desc=f'Training Epoch {epoch} '):

    if isinstance(d['input_ids'],list):
      input_ids = torch.stack(d['input_ids'],1).to(device)
    else:
      input_ids = d['input_ids'].to(device)

    targets = d["label"].to(device)
    outputs = model(
      input_ids=input_ids,
    )

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



def train_epoch_multi(model, data_loader, loss_fn1, loss_fn2, optimizer,  scheduler, epoch=1):

  model = model.train()
  device = model.device
  losses = []
  mlm_losses = []
  predict_out = []
  all_label_ids = []

  correct_predictions = 0

  data = tqdm(data_loader, desc=f'Training Epoch {epoch} ')
  for d in data:
    data.set_description(f'Training Epoch {epoch} / mlm loss: {np.mean(mlm_losses):.4f}  / classification loss: {np.mean(losses):.4f}')
    input_ids = d['input_ids'].to(device)
    batch_task = max(d["task"])

    outputs = model(
      input_ids=input_ids,
      task=batch_task
    )
    targets = d["label"].to(device)

    if batch_task == "mlm":
      
      loss = loss_fn1(outputs.transpose(1,2), targets)
      mlm_losses.append(loss.item())

    else:
     
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn2(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
    
      predict_out.extend(preds.tolist())
      all_label_ids.extend(targets.tolist())

    loss.backward()

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
      if isinstance(d['input_ids'],list):
        input_ids = torch.stack(d['input_ids'],1).to(device)
      else:
        input_ids = d['input_ids'].to(device)

      targets = d["label"].to(device)
      outputs = model(
        input_ids=input_ids,
      )

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
      if isinstance(d['input_ids'],list):
        input_ids = torch.stack(d['input_ids'],1).to(device)
      else:
        input_ids = d['input_ids'].to(device)

      outputs = model(
        input_ids=input_ids,
      )
      _, preds = torch.max(outputs, dim=1)

      predict_out.extend(preds.tolist())

  return predict_out
