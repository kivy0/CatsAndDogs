import torch
from sklearn.metrics import confusion_matrix
import numpy as np


def test(model, test_loader, device):
    correct = 0
    total = 0
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            #try:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            #except:
            #print(images)
            predictions, _ = torch.max(outputs, 1)
            predictions = torch.round(predictions)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            y_true = np.concatenate((y_true, labels.cpu().numpy()))
            y_pred = np.concatenate((y_pred, predictions.cpu().numpy()))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  
        print(f'tn: {tn} , fp: {fp} , fn: {fn} , tp: {tp}')   
        print(f'Accuracy: {100 * correct / total} %')

    return total 

