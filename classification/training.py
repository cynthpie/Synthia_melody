import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from model import MyResNet
from torch.utils.data import DataLoader, Dataset, random_split

USE_GPU = True
dtype = torch.float32 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)
    
def get_args():
    parser = argparse.ArgumentParser(description='ResNet18')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay [default: 0.0]')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv2D in ResNet18 [default: 64]')
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size [default: 16]')
    args = parser.parse_args()
    return args
    

print_every = 50
def check_accuracy(loader, model, analysis=False, logging=True, logfile=None):
    # function for test accuracy on validation and test set
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x[:, None, :, :] # FIX THIS LATER
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            if t == 0 and analysis:
              stack_labels = y
              stack_predicts = preds
            elif analysis:
              stack_labels = torch.cat([stack_labels, y], 0)
              stack_predicts = torch.cat([stack_predicts, preds], 0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct of val set (%.2f)' % (num_correct, num_samples, 100 * acc))
        if logging:
            logfile.write('Got %d / %d correct of val set (%.2f) \n' % (num_correct, num_samples, 100 * acc))
        return float(acc)

def train_part(model, optimizer, epochs=1, logging=True, logfile=None):
    """
    Train a model on NaturalImageNet using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x[:, None, :, :] # FIX THIS LATER
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
            if logging:
                logfile.write('Epoch: %d, Iteration %d, loss = %.4f \n' % (e, t, loss.item()))
        check_accuracy(loader_val, model, logging=logging, logfile=logfile)
                
if __name__== "__main__":

    # load data
    print("before data loaded")
    train_ds = torch.load("train_data24.pt")
    val_ds = torch.load("val_data24.pt")
    print("data loaded")
    # define dataloader
    loader_train = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    loader_val = DataLoader(val_ds, batch_size=args.batch, shuffle=True)
    
    print(next(iter(loader_train)))
    # define and train the network
    model = MyResNet()
    # print(model)
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.wd) 

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {} \n".format(params))

    logging = True
    if logging:
        logfile_name = "logfile24.txt"
        logfile = open(logfile_name, "w")
        logfile.write("Total number of model parameters is: {} \n".format(params))
        logfile.write("Training data size is: {} \n".format(len(train_ds)))
    else:
        logfile = None

    train_part(model, optimizer, epochs = 20, logging=logging, logfile=logfile)
    logfile.close()

    # # report test set accuracy
    # check_accuracy(loader_val, model, analysis=False)