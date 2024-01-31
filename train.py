import torch
import torch.optim as optim
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
import copy


def evaluate(model, eval_loader, loss_fn, device):
    model = model.eval()
    r_loss = 0.0
    preds = []
    ans = []

    for batch in tqdm(eval_loader):
        x_batch = batch['image'].to(device)
        y_batch = batch['label'].to(device)

        with torch.inference_mode():
            y_pred = model(x_batch)
            _, predicted = torch.max(y_pred.data, 1)

            loss = loss_fn(y_pred.to(device), y_batch)
            r_loss += loss.item()

            preds.extend(predicted.cpu().numpy())
            ans.extend(y_batch.cpu().numpy())

    return f1_score(ans, preds, average='micro'), r_loss / len(eval_loader)
    


def train_model(model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs):
    train_loss_history = []
    val_loss_history = []
    train_f1_history = []
    val_f1_history = []
    cur_f1 = 0

    for ep in range(num_epochs):
        model = model.train()
        running_loss = 0.0
        preds_train = []
        ans_train = []

        for i, batch in enumerate(train_loader):

            x_batch = batch['image'].to(device)
            y_batch = batch['label'].to(device)

            optimizer.zero_grad()

            y_pred = model(x_batch)
            _, predicted = torch.max(y_pred.data, 1)
            loss = loss_fn(y_pred.to(device), y_batch)

            
            print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                            ep+1,
                            num_epochs,
                            i * len(y_batch),
                            len(train_loader.dataset),
                            100. * i / len(train_loader),
                            loss.cpu().data.item()),
                            end='')

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            preds_train.extend(predicted.cpu().numpy())
            ans_train.extend(y_batch.cpu().numpy())

        train_f1 = f1_score(ans_train, preds_train, average='micro')
        train_loss = running_loss / len(train_loader)
        print(f'\navg train loss:{train_loss:.3f}, f1 on train:{train_f1:.3f}')
        train_loss_history.append(train_loss)
        train_f1_history.append(train_f1)

        val_f1, val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f'avg val loss:{val_loss:.3f}, f1 on validation:{val_f1:.3f}\n')
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)

        if val_f1 > cur_f1:
            cur_f1 = val_f1
            best_model_wts = copy.deepcopy(model)

    return best_model_wts, train_loss_history, train_f1_history, val_loss_history, val_f1_history
        