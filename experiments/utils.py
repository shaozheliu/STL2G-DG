from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, roc_auc_score
import torch


def test_evaluate(model, device, X, y, num_class):
    inputs = torch.from_numpy(X).type(torch.cuda.FloatTensor).to(device)
    labels = torch.from_numpy(y).type(torch.cuda.FloatTensor).to(device)
    outputs, domain_outputs = model(inputs, 0.1)
    # te = torch.softmax(outputs, dim=1)
    proba, preds = torch.max(torch.sigmoid(outputs), 1)
    if num_class > 2:
        te = torch.softmax(outputs, dim=1)
    else:
        te = torch.softmax(outputs, 1)[:, 1]
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    te = te.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    ka = cohen_kappa_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    if num_class > 2:
        roc_auc = roc_auc_score(labels, te, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(labels, te)
    return acc, ka, prec, recall, roc_auc

def test_evaluate_base(model, device, X, y, num_class):
    inputs = torch.from_numpy(X).type(torch.cuda.FloatTensor).to(device)
    labels = torch.from_numpy(y).type(torch.cuda.FloatTensor).to(device)
    outputs = model(inputs)
    # te = torch.softmax(outputs, dim=1)
    proba, preds = torch.max(torch.sigmoid(outputs), 1)
    if num_class > 2:
        te = torch.softmax(outputs, dim=1)
    else:
        te = torch.softmax(outputs, 1)[:, 1]
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    te = te.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    ka = cohen_kappa_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    if num_class > 2:
        roc_auc = roc_auc_score(labels, te, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(labels, te)
    return acc, ka, prec, recall, roc_auc