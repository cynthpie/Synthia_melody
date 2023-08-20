print("beginnign of script")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
import os
from sampleCNN import MySampleCNN
from get_data import get_dataset
from earlystopper import EarlyStopper
import wandb
from torch.utils.data import DataLoader, Dataset, Subset, random_split, TensorDataset
print("package imported ")

USE_GPU = True
dtype = torch.float32 
print("torch.cuda.is_available()", torch.cuda.is_available())
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)

def get_data_args(shift_type=None, shift_strength=0.0, shift_way=None, waveshape=None):
    parser = argparse.ArgumentParser(description='data_args')
    parser.add_argument('--shift_type', type=str, default=shift_type,
                        help='type of dataset shift [default: None]')
    parser.add_argument('--shift_strength', type=float, default=shift_strength,
                        help='strength of dataset shift [default: 0.0]')
    parser.add_argument('--shift_way', type=list, default=shift_way,
                        help='way to construct shift. [default: None]')
    parser.add_argument('--waveshape', type=str, default=waveshape,
                        help='waveshape made up of the data if no shift. [default: None]')
    args = parser.parse_args()
    return args
    
def get_training_args():
    parser = argparse.ArgumentParser(description='training_args')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay [default: 0.0]')
    parser.add_argument('--batch', type=int, default=64,
                        help='batch size [default: 64]')
    parser.add_argument("--max_epoch", type=int, default=15,
                        help="max number of epoch for training")
    parser.add_argument("--min_epoch", type=int, default=10,
                        help="min number of epoch for training")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="optimizer used in training")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of class labels [default: 2]")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="maximum magnitude of gradient updated [default: 2]")
    parser.add_argument("--patience", type=int, default=2,
                        help="patience for early stopper")
    parser.add_argument("--min_delta", type=float, default=0.01,
                        help="min_delta for early stopper")
    parser.add_argument("--seed", type=int, default=55555,
                        help="torch seed to train model")
    args = parser.parse_args()
    return args

def get_model_args():
    parser = argparse.ArgumentParser(description='model_args')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv1D in SampleCNN [default: 64]')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of class labels [default: 2]")
    parser.add_argument("--remove_classifier", type=bool, default=False,
                        help="remove last convolution layer in the model for DANN [default: 2]")
    args = parser.parse_args()
    return args

def load_data():
        sine_train_ds = torch.load("sine_train_data.pt") 
        sine_val_ds = torch.load("sine_val_data.pt")
        sine_test_ds = torch.load("sine_test_data.pt")
        square_train_ds = torch.load("square_train_data.pt") 
        square_val_ds = torch.load("square_val_data.pt")
        square_test_ds = torch.load("square_test_data.pt")
        triangle_train_ds = torch.load("triangle_train_data.pt") 
        triangle_val_ds = torch.load("triangle_val_data.pt")
        triangle_test_ds = torch.load("triangle_test_data.pt")
        sawtooth_train_ds = torch.load("sawtooth_train_data.pt") 
        sawtooth_val_ds = torch.load("sawtooth_val_data.pt")
        sawtooth_test_ds = torch.load("sawtooth_test_data.pt")
        return [sine_train_ds, sine_val_ds, sine_test_ds, square_train_ds, square_val_ds, square_test_ds, \
            triangle_train_ds, triangle_val_ds, triangle_test_ds, sawtooth_train_ds, sawtooth_val_ds, sawtooth_test_ds]

def load_and_log_data():
    with wandb.init(project="msc_project", job_type="load-data") as run:
        datasets = load_data()
        names = ["sine_train_ds", "sine_val_ds", "sine_test_ds", "square_train_ds", "square_val_ds", "square_test_ds", \
            "triangle_train_ds", "triangle_val_ds", "triangle_test_ds", "sawtooth_train_ds", "sawtooth_val_ds", "sawtooth_test_ds"]
        raw_data = wandb.Artifact(
            name="wav_dataset_exp001", type="dataset",
            description="dataset used to run experiment 001",
            metadata={"source": "generated by parallel_data_generation 001",
                      "sizes":[len(dataset) for dataset in datasets]})
        print("finish create artifact")
        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                if isinstance(data, torch.utils.data.dataset.Subset):
                    data = data.dataset
                x, y = data.tensors
                print(x.size(), y.size())
                torch.save(TensorDataset(x,y), file)
                print(f"finish saving {name}")
        print("finish add files to artifact")
        run.log_artifact(raw_data)
        print("finish logging")
        return None

def build_and_log_model(model_args):
    with wandb.init(project="msc_project", job_type="initialize", config=model_args) as run:
        model = MySampleCNN(model_args)
        model_artifact = wandb.Artifact(
            name="SampleCNN", type="model",
            description="SampleCNN for waveform data", 
            metadata=vars(model_args))
        torch.save(model.state_dict(), "initialized_model.pth")
        model_artifact.add_file("initialized_model.pth")
        run.log_artifact(model_artifact)
    return None

def train(model, train_loader, val_loader, args):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    model.train()
    example_ct = 0
    if args.num_classes==2:
        loss_fcn = nn.BCELoss()
    else:
        loss_fcn = F.cross_entropy
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.wd)
    earlystopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    wandb.watch(model, loss_fcn, log="all", log_freq=1000)
    for e in range(args.max_epoch):
        print("this is epoch", e)
        for t, (x, y) in enumerate(train_loader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype) if args.num_classes==2 else y.to(device=device, dtype=torch.long) 
            scores = torch.squeeze(model(x)) # since model is full-conv
            loss = loss_fcn(scores, y)
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters of the model using the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            example_ct += len(x)
            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                train_log(loss, example_ct, e)
        val_loss, val_accuracy, val_outputs = test(model, val_loader, args)
        test_log(val_loss, val_accuracy, example_ct, e)
        if earlystopper.early_stop(val_loss) and e >= args.min_epoch: 
            break

def train_and_log(train_args, model_args, data_args):
    with wandb.init(project="msc_project", job_type=f"{data_args.shift_type}_train", config=train_args) as run:
        config = wandb.config
        data = run.use_artifact('wav_dataset_exp001:latest')
        data_dir = data.download()
        data_dir = data_dir[2:]
        train_ds = get_dataset(data_dir, "train", shift_type=data_args.shift_type, waveshape=data_args.waveshape,\
                    shift_strength=data_args.shift_strength, shift_way=data_args.shift_way)
        val_ds = get_dataset(data_dir, "val", shift_type=data_args.shift_type, waveshape=data_args.waveshape,\
                    shift_strength=data_args.shift_strength, shift_way=data_args.shift_way)
        data_config = vars(data_args)
        config.update(data_config)
        print("finish load data from artifact")

        train_loader = DataLoader(train_ds, batch_size=train_args.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=train_args.batch, shuffle=True)
        print("finish initialise data loader")

        model_artifact = run.use_artifact("SampleCNN:latest")
        print("finish use model artifact")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir[2:], "initialized_model.pth")
        model_config = model_artifact.metadata
        config.update(model_config)
        args = argparse.Namespace(**config) 
        model = MySampleCNN(model_args)
        print("finish create model")
        print("model_dir=", model_dir)
        print("model_path=", model_path)
        model.load_state_dict(torch.load(model_path))
        train(model, train_loader, val_loader, train_args)

        model_artifact = wandb.Artifact(
            "trained-sampleCNN", type="model",
            description="Trained SampleCNN model",
            metadata=dict(model_config))
        torch.save(model.state_dict(), "trained_model.pth")
        model_artifact.add_file("trained_model.pth")
        wandb.save("trained_model.pth")
        run.log_artifact(model_artifact)
    return model

def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    num_correct = 0
    num_samples = 0
    outputs = None
    if args.num_classes==2:
        loss_fcn = nn.BCELoss(reduction='sum')
    else:
        loss_fcn = F.cross_entropy
    with torch.no_grad():
        for t, (x, y) in enumerate(test_loader):
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=dtype) if args.num_classes==2 else y.to(device=device, dtype=torch.long) 
            scores = torch.squeeze(model(x))
            if outputs is None:
                outputs = scores
            else:
                outputs = torch.cat((outputs, scores))
            if args.num_classes==2:
                test_loss += loss_fcn(scores, y)
                preds = scores>0.5
            else:
                test_loss += loss_fcn(scores, y)
                _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            print("scores", scores)
            print("preds:", preds, "y: ",y)
            print("num_correct:", num_correct, "num_samples:", num_samples)
    test_loss /= len(test_loader.dataset)
    acc = float(num_correct) / num_samples
    return test_loss, float(acc), outputs

def evaluate(model, test_loader, args):
    loss, accuracy, outputs = test(model, test_loader, args)
    return loss, accuracy, outputs

def evaluate_and_log(data_args, model_args=None):
    with wandb.init(project="msc_project", job_type=f"{data_args.shift_type}_eval", config=model_args) as run:
        config = wandb.config
        data_config = vars(data_args)
        config.update(data_config)
        data = run.use_artifact('wav_dataset_exp001:latest')
        data_dir = data.download()
        data_dir = data_dir[2:]
        same_dist_testdata = get_dataset(data_dir, "test", shift_type=data_args.shift_type, shift_strength=data_args.shift_strength,\
            shift_way=data_args.shift_way)
        anti_biased_testdata = get_dataset(data_dir, "test", shift_type=data_args.shift_type, shift_strength=data_args.shift_strength,\
            shift_way=(data_args.shift_way)[::-1])
        neutral_testdata = get_dataset(data_dir, "test", shift_type="domain_shift", shift_strength=0.5,\
            shift_way=(data_args.shift_way))
        sine_test_data = get_dataset(data_dir, "test", waveshape="sine")
        square_test_data = get_dataset(data_dir, "test", waveshape="square")
        sawtooth_test_data = get_dataset(data_dir, "test", waveshape="sawtooth")
        triangle_test_data = get_dataset(data_dir, "test", waveshape="triangle")

        batch_size = 64
        same_dist_loader = DataLoader(same_dist_testdata, shuffle=False, batch_size=batch_size)
        anti_biased_loader = DataLoader(anti_biased_testdata, shuffle=False, batch_size=batch_size)
        neutral_dist_loader = DataLoader(neutral_testdata, shuffle=False, batch_size=batch_size)
        sine_test_loader = DataLoader(sine_test_data, shuffle=False, batch_size=batch_size)
        sawtooth_test_loader = DataLoader(sawtooth_test_data, shuffle=False, batch_size=batch_size)
        square_test_loader = DataLoader(square_test_data, shuffle=False, batch_size=batch_size)
        triangle_test_loader = DataLoader(triangle_test_data, shuffle=False, batch_size=batch_size)

        model_artifact = run.use_artifact("trained-sampleCNN:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pth")
        model_config = model_artifact.metadata
        config.update(model_config)
        args = argparse.Namespace(**model_config) 

        model = MySampleCNN(args)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        same_dist_loss, same_dist_accuracy, same_dist_outputs = evaluate(model, same_dist_loader, args)
        anti_biased_loss, anti_biased_accuracy, anti_biased_outputs = evaluate(model, anti_biased_loader, args)
        neutral_loss, neutral_accuracy, neutral_outputs = evaluate(model, neutral_dist_loader, args)
        sine_loss, sine_accuracy, sine_outputs = evaluate(model, sine_test_loader, args)
        sawtooth_loss, sawtooth_accuracy, sawtooth_outputs = evaluate(model, sawtooth_test_loader, args)
        triangle_loss, triangle_accuracy, triangle_outputs = evaluate(model, triangle_test_loader, args)
        square_loss, square_accuracy, square_outputs = evaluate(model, square_test_loader, args)

        same_dist_outputs = same_dist_outputs.numpy(force=True)
        anti_biased_outputs = anti_biased_outputs.numpy(force=True)
        neutral_outputs = neutral_outputs.numpy(force=True)
        sine_outputs = sine_outputs.numpy(force=True)
        sawtooth_outputs = sawtooth_outputs.numpy(force=True)
        triangle_outputs = triangle_outputs.numpy(force=True)
        square_outputs = square_outputs.numpy(force=True)      

        np.savetxt('same_dist_outputs.txt', same_dist_outputs, fmt='%1.3f')
        np.savetxt('anti_biased_outputs.txt', anti_biased_outputs, fmt='%1.3f')
        np.savetxt('neutral_outputs.txt', neutral_outputs, fmt='%1.3f')
        np.savetxt('sine_outputs.txt', sine_outputs, fmt='%1.3f')
        np.savetxt('sawtooth_outputs.txt', sawtooth_outputs, fmt='%1.3f')
        np.savetxt('triangle_outputs.txt', triangle_outputs, fmt='%1.3f')
        np.savetxt('square_outputs.txt', square_outputs, fmt='%1.3f')
        run.summary.update({"sine_loss": sine_loss, "sine_accuracy": sine_accuracy,
                            "sawtooth_loss":sawtooth_loss, "sawtooth_accuracy":sawtooth_accuracy,
                            "square_loss":square_loss, "square_accuracy":square_accuracy,
                            "triangle_loss": triangle_loss, "triangle_accuracy":triangle_accuracy,
                            "anti_bias_loss": anti_biased_loss, "anti_bias_accuracy":anti_biased_accuracy,
                            "neutral_loss": neutral_loss, "neutral_accuracy":neutral_accuracy,
                            "same_dist_loss": same_dist_loss, "same_dist_accuracy":same_dist_accuracy})
        wandb.save("sine_outputs.txt")
        wandb.save("sawtooth_outputs.txt")
        wandb.save("triangle_outputs.txt")
        wandb.save("square_outputs.txt")
        wandb.save("same_dist_outputs.txt")
        wandb.save("anti_biased_outputs.txt")
        wandb.save("neutral_outputs.txt")
    return None

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")

if __name__== "__main__":
    print_every = 50
    train_args = get_training_args()
    model_args = get_model_args()
    torch.cuda.manual_seed(train_args.seed) # 5, 55, 555, 5555, 55555
    # load_and_log_data() # use if have new data
    # build_and_log_model(model_args) # use if have new model
    train_data_len = 40000
    strengths = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for strength in strengths:
        train_data_args = get_data_args(shift_type="sample_selection_bias", shift_strength=strength, shift_way=["sine", "square"],\
            waveshape=None)
        train_and_log(train_args, model_args, train_data_args)
        evaluate_and_log(train_data_args, None)