import argparse
from datetime import datetime
import logging
import torch
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from models.datasets import get_dataloader, get_dataloader_eval, get_dataloader_multi
from models.multitask_model import MultiTaskModel
from models.vanilla_bert import VanillaBERT
from models.bert_add_layer import TestBERT

from tqdm import tqdm
from utils import make_submission, set_seed
from pathlib import Path
import pandas as pd

from utils.train import TrainInfo, eval_model, predict, train_epoch, train_epoch_multi

logging.basicConfig(format="%(asctime)s  [%(levelname)s]  %(message)s")
myLogger = logging.getLogger()

def main(args):
    # set seed
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("WARNING: Running on CPU. This will be slow.")


    if args.action == 'split':
        print('Splitting data...')

        df = pd.read_csv(args.train_file)
        output_dir = Path(args.output_dir)

        df_train = df.sample(frac=args.split_size, random_state=args.seed)
        df_test = df.drop(df_train.index)

        df_train.to_csv(output_dir / "train.csv", index=False)
        df_test.to_csv(output_dir / "valid.csv", index=False)
        print("Train/Test split complete. Train size: {}, Test size: {}".format(len(df_train), len(df_test)))
        return

    if args.action == 'train-multi':
        myLogger.info("Training multitask model...")

        tokenizer = MultiTaskModel.get_tokenizer(args.model)

        dloader_train = get_dataloader_multi(args.train_file, args.mlm_file,
                             tokenizer, args.batch_size, args.max_len)
        
        if args.test_file:
            dloader_valid = get_dataloader(args.test_file, args.task, tokenizer, args.batch_size_eval, args.max_len, shuffle=False)

        dloader_eval = get_dataloader_eval('a', tokenizer, args.batch_size_eval, args.max_len)

        model = MultiTaskModel(n_classes=2, dropout=args.dropout, pre_trained_model=args.model).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        loss_fn1 = torch.nn.CrossEntropyLoss().to(device)
        loss_fn2 = torch.nn.CrossEntropyLoss().to(device)

        scheduler = get_linear_schedule_with_warmup(
                        optimizer=optimizer, 
                        num_warmup_steps=args.warmup_steps,
                        num_training_steps=len(dloader_train)*args.epochs,
                    )

        patience = 2
        delta = 1e-2

        last_f1 = 0     
        failed = 0    
        best_model_file = Path(args.output_dir) / datetime.now().strftime('%Y%m%d-%H%M%S/multi_best_model_state.bin')
        best_model_file.parent.mkdir(parents=True, exist_ok=True)
        # log training results to file
        fileHandler = logging.FileHandler(best_model_file.parent / 'train.log')
        myLogger.addHandler(fileHandler)

        
        info = TrainInfo()
        
        for epoch in range(1, args.epochs+1):

            train_acc, train_f1, train_rep, train_loss = train_epoch_multi(model, dloader_train, loss_fn1, loss_fn2,
                     optimizer, scheduler, epoch)
            myLogger.info(f'Epoch {epoch} train report: \n\n {train_rep}')
            info.add_train_info(train_acc, train_f1, train_loss)
            if args.test_file:
                acc, f1, rep, loss = eval_model(model, dloader_valid, loss_fn2)
                myLogger.info(f'Epoch {epoch} eval report: \n\n {rep}')

                if info.best_f1<=f1:
                    failed = 0
                    info.best_f1 = f1
                    info.best_acc = acc
                    info.best_loss = loss
                    myLogger.info(f'New best model found. Saving to {best_model_file.parent.name}')
                    torch.save(model.state_dict(), best_model_file)
                else:
                    if f1<last_f1 and abs(f1-last_f1)>delta:
                        failed += 1 if abs(f1-last_f1)>delta else 0
                        if failed > patience: 
                            myLogger.info(f'Patience {failed} reached. Stopping training...')
                            break
                        myLogger.info(f'Metric falling, now at patience {failed}')
                    else:
                        failed = 0
                last_f1 = f1

        model.load_state_dict(torch.load(best_model_file, map_location=device))

        myLogger.info(f'Starting prediction for task {args.task}')

        predictions = predict(model, dloader_eval)

        make_submission(args.task, predictions, best_model_file.parent)

        myLogger.info(f'Finished prediction for task {args.task}')

        
    if args.action == 'train-bert':
        myLogger.info("Training model...")

        tokenizer = VanillaBERT.get_tokenizer(args.model)        

        dloader_train = get_dataloader(args.train_file, args.task, tokenizer, args.batch_size, args.max_len, shuffle=True)
        if args.test_file:
            dloader_valid = get_dataloader(args.test_file, args.task, tokenizer, args.batch_size_eval, args.max_len, shuffle=False)

        dloader_eval = get_dataloader_eval(args.task, tokenizer, args.batch_size_eval, args.max_len)

        if args.task == 'a':
            num_classes = 2
        elif args.task == 'b':
            num_classes = 4
        elif args.task == 'c':
            num_classes = 11

        model = TestBERT(n_classes=num_classes, dropout=args.dropout, pre_trained_model=args.model).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        scheduler = get_linear_schedule_with_warmup(
                        optimizer=optimizer, 
                        num_warmup_steps=args.warmup_steps,
                        num_training_steps=len(dloader_train)*args.epochs,
                    )

        patience = 2
        delta = 1e-2

        last_f1 = 0     
        failed = 0    
        best_model_file = Path(args.output_dir) / datetime.now().strftime('%Y%m%d-%H%M%S/best_model_state.bin')
        best_model_file.parent.mkdir(parents=True, exist_ok=True)
        # log training results to file
        fileHandler = logging.FileHandler(best_model_file.parent / 'train.log')
        myLogger.addHandler(fileHandler)
        
        info = TrainInfo()
        
        for epoch in range(1, args.epochs+1):
            train_acc, train_f1, train_rep, train_loss = train_epoch(model, dloader_train, loss_fn, optimizer, scheduler, epoch)
            myLogger.info(f'Epoch {epoch} train report: \n\n {train_rep}')
            info.add_train_info(train_acc, train_f1, train_loss)
            if args.test_file:
                acc, f1, rep, loss = eval_model(model, dloader_valid, loss_fn)
                myLogger.info(f'Epoch {epoch} eval report: \n\n {rep}')

                if info.best_f1<=f1:
                    failed = 0
                    info.best_f1 = f1
                    info.best_acc = acc
                    info.best_loss = loss
                    myLogger.info(f'New best model found. Saving to {best_model_file.parent.name}')
                    torch.save(model.state_dict(), best_model_file)
                else:
                    if f1<last_f1 and abs(f1-last_f1)>delta:
                        failed += 1 if abs(f1-last_f1)>delta else 0
                        if failed > patience: 
                            myLogger.info(f'Patience {failed} reached. Stopping training...')
                            break
                        myLogger.info(f'Metric falling, now at patience {failed}')
                    else:
                        failed = 0
                last_f1 = f1

        model.load_state_dict(torch.load(best_model_file, map_location=device))

        myLogger.info(f'Starting prediction for task {args.task}')

        predictions = predict(model, dloader_eval)

        make_submission(args.task, predictions, best_model_file.parent)

        myLogger.info(f'Finished prediction for task {args.task}')

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['train-bert', 'train-multi', 'evaluate', 'split'], default='train-bert')
    parser.add_argument('--split-size', type=float, default=0.8)
    parser.add_argument('--task', choices=['a', 'b', 'c'], default='a')
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-eval', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max-len', type=int, default=64)
    parser.add_argument('--train-file', type=str, default='data/train.csv')
    parser.add_argument('--test-file', type=str, default='data/test.csv')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--mlm-file', type=str, default='data/gab_1M_unlabelled.csv')
    parser.add_argument('--skip-predict', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup-steps', type=int, default=0)
    args = parser.parse_args()

    main(args)