import argparse
from datetime import datetime
import logging
import torch
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from models.datasets import get_dataloader, get_dataloader_eval
from models.embeddings_bert import EmbeddingsBERT
from models.vanilla_bert import VanillaBERT

from tqdm import tqdm
from utils import make_submission, set_seed, task_to_classes
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.train import EarlyStopper, TrainInfo, eval_model, predict, train_epoch

formatter = logging.Formatter(fmt="%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
myLogger = logging.getLogger()
myLogger.setLevel(logging.INFO)
# myLogger both prints to console and writes to file
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)
myLogger.addHandler(consoleHandler)



def main(args):
    # set seed
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print("WARNING: Running on CPU. This will be slow.")


        
    if args.action in ['train-bert','train-bert-embeddings']:
        myLogger.info("Training model...")

        df = pd.read_csv(args.train_file)
        df_emb = None
        if args.embeddings_file:
            df_emb = pd.read_csv(args.embeddings_file)
    

        df_train, df_valid = train_test_split(df, test_size=0.1, random_state=args.seed)
        dloader_valid = get_dataloader(df_valid, args.task, args.batch_size_eval, args.max_len, shuffle=False, df_emb=df_emb)
        dloader_test = None
        dloader_eval = None

        if not args.skip_test_split:
            df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=args.seed)
            dloader_test = get_dataloader(df_test, args.task,  args.batch_size_eval, args.max_len, shuffle=False, df_emb=df_emb)
        
        dloader_train = get_dataloader(df_train, args.task,  args.batch_size, args.max_len, shuffle=True, df_emb=df_emb)
        

        if args.eval_file:
            dloader_eval = get_dataloader_eval(args.task,  args.batch_size_eval, args.max_len)

        num_classes = task_to_classes(args.task)

        if args.action == 'train-bert':        
            model = VanillaBERT(n_classes=num_classes, dropout=args.dropout, pre_trained_model=args.model).to(device)
        else:
            assert args.embeddings_file, "Embeddings file must be provided for training embeddings model"
            model = EmbeddingsBERT(n_classes=num_classes, dropout=args.dropout, pre_trained_model=args.model).to(device)            

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        # loss_fn = torch.nn.NLLLoss().to(device)
        scheduler = get_linear_schedule_with_warmup(
                        optimizer=optimizer, 
                        num_warmup_steps=args.warmup_steps,
                        num_training_steps=len(dloader_train)*args.epochs,
                    )


        early_stopper = EarlyStopper(patience=2, min_delta=1e-3)

        best_model_file = Path(args.output_dir) / datetime.now().strftime('%Y%m%d-%H%M%S/best_model_state.bin')
        best_model_file.parent.mkdir(parents=True, exist_ok=True)
        # log training results to file
        fileHandler = logging.FileHandler(best_model_file.parent / 'train.log')
        fileHandler.setFormatter(formatter)
        myLogger.addHandler(fileHandler)
        
        info = TrainInfo()
        
        for epoch in range(1, args.epochs+1):
            train_acc, train_f1, train_rep, train_loss = train_epoch(model, dloader_train, loss_fn, optimizer, scheduler, epoch)
            myLogger.info(f'Epoch {epoch} train report: \n\n {train_rep}')
            info.add_train_info(train_acc, train_f1, train_loss)

            acc, f1, rep, loss = eval_model(model, dloader_valid, loss_fn)
            myLogger.info(f'Epoch {epoch} eval report: \n\n {rep}')

            if early_stopper.early_stop_loss(loss):
                myLogger.info(f'Patience {early_stopper.patience} reached. Stopping training...')
                break
            
            if info.best_loss>loss:
                info.best_loss = loss
                info.best_acc = acc
                info.best_f1 = f1
                torch.save(model.state_dict(), best_model_file)
                myLogger.info(f'New best model found. Saving to {best_model_file.parent.name}')
            else:
                myLogger.info(f'Metric falling, now at patience {early_stopper.counter} at epoch {epoch}...')

        myLogger.info(f'Finished training. Best model was saved to {best_model_file.parent.name}')

        model.load_state_dict(torch.load(best_model_file, map_location=device))

        if not args.skip_predict:
            myLogger.info(f'Starting prediction for task {args.task}')
            eval_model(model, dloader_test,loss_fn)


        if args.eval_file:
            predictions = predict(model, dloader_eval)
            make_submission(args.task, predictions, best_model_file.parent)
            myLogger.info(f'Finished prediction for task {args.task}')

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['train-bert', 'train-bert-embeddings', 'evaluate', 'split'], default='train-bert')
    parser.add_argument('--split-size', type=float, default=0.8)
    parser.add_argument('--task', choices=['a', 'b'], default='a')
    parser.add_argument('--model', type=str, default='dbmdz/bert-base-italian-cased')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-eval', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max-len', type=int, default=96)
    parser.add_argument('--train-file', type=str, default='data/subtaskA_train.csv')
    parser.add_argument('--skip-test-split', action='store_true')
    parser.add_argument('--eval-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--embeddings-file', type=str, default=None)
    parser.add_argument('--skip-predict', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup-steps', type=int, default=0)
    args = parser.parse_args()

    main(args)