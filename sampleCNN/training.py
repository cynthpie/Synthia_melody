print("beginnign of script")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import os
from sampleCNN import MySampleCNN
import wandb
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
print("package imported ")

USE_GPU = True
dtype = torch.float32 
print("torch.cuda.is_available()", torch.cuda.is_available())
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)
torch.cuda.manual_seed(5)

def get_training_args():
    parser = argparse.ArgumentParser(description='training_args')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay [default: 0.0]')
    parser.add_argument('--batch', type=int, default=64,
                        help='batch size [default: 16]')
    parser.add_argument("--epochs", type=int, default=8,
                        help="number of epoch for training")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="optimizer used in training")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of class labels [default: 2]")
    args = parser.parse_args()
    return args

def get_model_args():
    parser = argparse.ArgumentParser(description='model_args')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv1D in SampleCNN [default: 64]')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of class labels [default: 2]")
    args = parser.parse_args()
    return args

def load_data():
        unbiased_train_ds = torch.load("unbiased_train_wav.pt") 
        unbiased_val_ds = torch.load("unbiased_val_wav.pt")
        unbiased_test_ds = torch.load("unbiased_test_wav.pt")
        biased_train_ds = torch.load("biased_train_wav.pt")
        biased_val_ds = torch.load("biased_val_wav.pt")
        anti_biased_test_ds = torch.load("anti-biased_test_wav.pt")
        neutral_test_ds = torch.load("neutral_test_wav.pt")
        square_test_ds = torch.load("square_test_wav.pt")
        triangle_test_ds = torch.load("triangle_test_wav.pt")
        return [unbiased_train_ds, unbiased_val_ds, unbiased_test_ds, biased_train_ds, biased_val_ds, anti_biased_test_ds, neutral_test_ds, square_test_ds, triangle_test_ds]

def load_and_log_data():
    with wandb.init(project="msc_project", job_type="load-data") as run:
        datasets = load_data()
        names = ["unbiased_train", "unbiased_val", "unbiased_test", "biased_train", "biased_val", "anti-biased_test", "neutral_test", "square_test", "triangle_test"]
        raw_data = wandb.Artifact(
            name="wav_dataset_exp001", type="dataset",
            description="waveform dataset used to run experiment 001",
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
    wandb.watch(model, loss_fcn, log="all", log_freq=1000)
    for e in range(args.epochs):
        print("this is epoch", e)
        for t, (x, y) in enumerate(train_loader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype) if args.num_classes==2 else y.to(device=device, dtype=torch.long) 
            scores = torch.squeeze(model(x), dim=-1) # since model is full-conv
            loss = loss_fcn(scores, y)
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters of the model using the gradients
            optimizer.step()
            example_ct += len(x)
            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                train_log(loss, example_ct, e)
        loss, accuracy = test(model, val_loader, args)
        test_log(loss, accuracy, example_ct, e)

def train_and_log(train_args, model_args):
    with wandb.init(project="msc_project", job_type="train", config=train_args) as run:
        config = wandb.config
        data = run.use_artifact('wav_dataset_exp001:latest')
        data_dir = data.download()
        data_dir = data_dir[2:]
        unbiased_train_ds = torch.load(os.path.join(data_dir+"/biased_train.pt"))
        unbiased_val_ds = torch.load(os.path.join(data_dir+"/biased_val.pt"))
        #biased_train_ds = torch.load(os.path.join(data_dir+"/biased_train.pt"))
        #biased_val_ds = torch.load(os.path.join(data_dir+"/biased_val.pt"))
        print("finish load data from artifact")

        train_loader = DataLoader(unbiased_train_ds, batch_size=train_args.batch, shuffle=True)
        val_loader = DataLoader(unbiased_val_ds, batch_size=train_args.batch, shuffle=True)
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
    if args.num_classes==2:
        loss_fcn = nn.BCELoss(reduction='sum')
    else:
        loss_fcn = F.cross_entropy
    with torch.no_grad():
        for t, (x, y) in enumerate(test_loader):
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=dtype) if args.num_classes==2 else y.to(device=device, dtype=torch.long) 
            scores = torch.squeeze(model(x), dim=-1)
            # test_loss += loss_fcn(scores, y) # reduction=sum?
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
    return test_loss, float(acc)

def get_hardest_k_examples(model, testing_set, args, k=32):
    model.eval()
    loader = DataLoader(testing_set, 1, shuffle=False)
    # get the losses and predictions for each item in the dataset
    losses = None
    predictions = None
    if args.num_classes==2:
        loss_fcn = nn.BCELoss()
    else:
        loss_fcn = F.cross_entropy
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device=device, dtype=dtype) 
            target = target.to(device=device, dtype=dtype) if args.num_classes==2 else y.to(device=device, dtype=torch.long) 
            output = torch.squeeze(model(data), dim=-1) 
            loss = loss_fcn(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
            else:
                losses = torch.cat((losses, loss.view((1, 1))), 0)
                predictions = torch.cat((predictions, pred), 0)
    argsort_loss = torch.argsort(losses, dim=0)

    argsort_loss = argsort_loss.cpu()
    losses = losses.cpu()
    predictions = predictions.cpu()
    highest_k_losses = losses[argsort_loss[-k:]]
    hardest_k_examples = testing_set[argsort_loss[-k:]][0]
    true_labels = testing_set[argsort_loss[-k:]][1]
    predicted_labels = predictions[argsort_loss[-k:]]
    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels

def evaluate(model, test_loader, args):
    loss, accuracy = test(model, test_loader, args)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset, args)
    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def evaluate_and_log(wav_data_path, args=None):
    with wandb.init(project="msc_project", job_type="report", config=args) as run:
        data = run.use_artifact('wav_dataset_exp001:latest')
        data_dir = data.download()
        data_dir = data_dir[2:]
        unbiased_test = torch.load(os.path.join(data_dir+"/unbiased_test.pt"))
        anti_biased_test = torch.load(os.path.join(data_dir+"/anti-biased_test.pt"))
        neutral_test = torch.load(os.path.join(data_dir+"/neutral_test.pt"))
        square_test = torch.load(os.path.join(data_dir+"/square_test.pt"))
        triangle_test = torch.load(os.path.join(data_dir+"/triangle_test.pt"))

        unbiased_test_loader = DataLoader(unbiased_test, shuffle=False, batch_size=128)
        anti_biased_test_loader = DataLoader(anti_biased_test, shuffle=False, batch_size=128)
        neutral_test_loader = DataLoader(neutral_test, shuffle=False, batch_size=128)
        square_test_loader = DataLoader(square_test, shuffle=False, batch_size=128)
        triangle_test_loader = DataLoader(triangle_test, shuffle=False, batch_size=128)

        model_artifact = run.use_artifact("trained-sampleCNN:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_model.pth")
        model_config = model_artifact.metadata
        args = argparse.Namespace(**model_config) 

        model = MySampleCNN(args)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        wav_unbiased_test_path = "/rds/general/user/cl222/home/audio/unbiased_train"
        wav_anti_biased_test_path = "/rds/general/user/cl222/home/audio/anti-biased_test"
        wav_neutral_test_path = "/rds/general/user/cl222/home/audio/neutral_test"

        # test on unbaised, anti-biased, and neutral data
        print("start evaluating unbiased test set")
        unbiased_loss, unbiased_accuracy, unbiased_highest_losses, unbiased_hardest_examples, \
            unbiased_true_labels, unbiased_preds = evaluate(model, unbiased_test_loader, args)
        print("start evaluating anti-biased test set")
        anti_biased_loss, anti_biased_accuracy, anti_biased_highest_losses, anti_biased_hardest_examples, \
            anti_biased_true_labels, anti_biased_preds = evaluate(model, anti_biased_test_loader, args)
        print("start evaluating neutral test set")
        neutral_loss, neutral_accuracy, neutral_highest_losses, neutral_hardest_examples, \
            neutral_true_labels, neutral_preds = evaluate(model, neutral_test_loader, args)
        print("start evaluating square test set")
        square_loss, square_accuracy, square_highest_losses, square_hardest_examples, \
            square_true_labels, square_preds = evaluate(model, square_test_loader, args)
        print("start evaluating triangle test set")
        triangle_loss, triangle_accuracy, triangle_highest_losses, triangle_hardest_examples, \
            triangle_true_labels, triangle_preds = evaluate(model, triangle_test_loader, args)        

        run.summary.update({"unbiased_loss": unbiased_loss, "unbiased_accuracy": unbiased_accuracy,
                            "anti_biased_loss":anti_biased_loss, "anti_biased_accuracy":anti_biased_accuracy,
                            "neutral_loss":neutral_loss, "neutral_accuracy":neutral_accuracy,
                            "square_loss":square_loss, "square_accuracy":square_accuracy,
                            "triangle_loss": triangle_loss, "triangle_accuracy":triangle_accuracy})

        # wandb.log({"unbiased_high-loss-examples":
        #     [wandb.Audio(torch.squeeze(hard_example, dim=0).cpu().detach().numpy(), sample_rate=16000, caption=str(int(pred)) + "," +  str(int(label)))
        #      for hard_example, pred, label in zip(unbiased_hardest_examples, unbiased_preds, unbiased_true_labels)],
        #      "anti_biased_high-loss-examples":
        #      [wandb.Audio(torch.squeeze(hard_example,dim=0).cpu().detach().numpy(), sample_rate=16000, caption=str(int(pred)) + "," +  str(int(label)))
        #      for hard_example, pred, label in zip(anti_biased_hardest_examples, anti_biased_preds, anti_biased_true_labels)],
        #      "neutral_high-loss-examples":
        #      [wandb.Audio(torch.squeeze(hard_example,dim=0).cpu().detach().numpy(), sample_rate=16000, caption=str(int(pred)) + "," +  str(int(label)))
        #      for hard_example, pred, label in zip(neutral_hardest_examples, neutral_preds, neutral_true_labels)]})
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
    # load_and_log_data() # use if have new data
    # build_and_log_model(model_args) # use if have new model
    # train_and_log(train_args, model_args)
    evaluate_and_log(model_args)