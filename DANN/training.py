print("beginnign of script")
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
import os, sys
sys.path.append("/rds/general/user/cl222/home/msc_project/sampleCNN")
from dann import MySampleCNN
from get_data import get_dataset
from earlystopper import EarlyStopper
import wandb
from torch.utils.data import DataLoader, Dataset, Subset, random_split, TensorDataset
from dann import FeatureExtractor, Classifier, Discriminator
print("package imported ")

USE_GPU = True
dtype = torch.float32 
print("torch.cuda.is_available()", torch.cuda.is_available())
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)
print_every = 50

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
    
def get_training_args(a, a_upper, l):
    parser = argparse.ArgumentParser(description='training_args')
    parser.add_argument('--lr', type=float, default=l,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay [default: 0.0]')
    parser.add_argument('--wd_d', type=float, default=0.0001,
                        help='weight decay [default: 0.0]')
    parser.add_argument('--wd_c', type=float, default=0.0,
                        help='weight decay [default: 0.0]')
    parser.add_argument('--wd_e', type=float, default=0.0,
                        help='weight decay [default: 0.0]')
    parser.add_argument('--batch', type=int, default=32,
                        help='batch size [default: 64]')
    parser.add_argument("--max_epoch", type=int, default=40,
                        help="max number of epoch for training")
    parser.add_argument("--min_epoch", type=int, default=15,
                        help="min number of epoch for training")
    parser.add_argument("--optimizer_e", type=str, default="Adam",
                        help="optimizer used in training")
    parser.add_argument("--optimizer_c", type=str, default="Adam",
                        help="optimizer used in training")
    parser.add_argument("--optimizer_d", type=str, default="SGD",
                        help="optimizer used in training")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="momentum in SGD")
    parser.add_argument("--beta1", type=float, default=0.9,
                         help="beta in adam")
    parser.add_argument("--beta2", type=float, default=0.999,
                         help="beta in adam")
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
    parser.add_argument("--alpha", type=float, default=a,
                        help="torch seed to train model")
    parser.add_argument("--alpha_upper", type=float, default=a_upper,
                        help="torch seed to train model")
    parser.add_argument("--activ_fcn", type=str, default="leakyrelu",
                        help="torch seed to train model")
    parser.add_argument("--neg_slope", type=float, default=0.0,
                        help="torch seed to train model")
    parser.add_argument("--lr_d", type=float, default=l*2,
                        help="torch seed to train model")
    args = parser.parse_args()
    return args

def get_discriminator_args():
    parser = argparse.ArgumentParser(description='discriminator_args')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv1D in SampleCNN [default: 64]')
    parser.add_argument("--num_domain", type=int, default=2,
                        help="number of domain in training and testing data [default: 2]")
    parser.add_argument("--drop_out_rate_d", type=float, default=0.5,
                        help="number of class labels [default: 2]")
    parser.add_argument("--stronger_dis", type=bool, default=False,
                        help="number of class labels [default: 2]")
    parser.add_argument("--num_blocks_d", type=bool, default=1,
                        help="number of class labels [default: 2]")
    parser.add_argument("--num_blocks_d2", type=bool, default=2,
                        help="number of class labels [default: 2]")
    args = parser.parse_args()
    return args

def get_extractor_args():
    parser = argparse.ArgumentParser(description='extractor_args')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv1D in SampleCNN [default: 64]')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of class labels [default: 2]")
    parser.add_argument("--remove_classifier", type=bool, default=True,
                        help="remove last convolution layer in the model for DANN [default: 2]")
    parser.add_argument("--num_block_e", type=int, default=5,
                        help="remove last convolution layer in the model for DANN [default: 2]")
    args = parser.parse_args()
    return args

def get_classifier_args():
    parser = argparse.ArgumentParser(description='extractor_args')
    parser.add_argument('--foc', type=int, default=64,
                        help='out-channel of first Conv1D in SampleCNN [default: 64]')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of class labels [default: 2]")
    parser.add_argument("--drop_out_rate_c", type=float, default=0.2,
                        help="number of class labels [default: 2]")
    parser.add_argument("--stronger_clas", type=bool, default=False,
                        help="number of class labels [default: 2]")
    parser.add_argument("--num_blocks_c", type=bool, default=4,
                        help="number of class labels [default: 2]")
    parser.add_argument("--num_blocks_c2", type=bool, default=0,
                        help="number of class labels [default: 2]")
    args = parser.parse_args()
    return args

def build_and_log_model(extractor_args, classifier_args, discriminator_args):
    with wandb.init(project="msc_project", job_type="dann_initialize", config=extractor_args) as run:
        extractor = FeatureExtractor(extractor_args)
        extractor_artifact = wandb.Artifact(
            name="extractor", type="model",
            description="extractor of DANN", 
            metadata=vars(extractor_args))
        torch.save(extractor.state_dict(), "initialized_extractor.pth")
        extractor_artifact.add_file("initialized_extractor.pth")
        run.log_artifact(extractor_artifact)

        classifier = Classifier(classifier_args)
        classifier_artifact = wandb.Artifact(
            name="classifier", type="model",
            description="classifier of DANN", 
            metadata=vars(classifier_args))
        torch.save(classifier.state_dict(), "initialized_classifier.pth")
        classifier_artifact.add_file("initialized_classifier.pth")
        run.log_artifact(classifier_artifact)

        discriminator = Discriminator(discriminator_args)
        discriminator_artifact = wandb.Artifact(
            name="discriminator", type="model",
            description="discriminator of DANN", 
            metadata=vars(discriminator_args))
        torch.save(discriminator.state_dict(), "initialized_discriminator.pth")
        discriminator_artifact.add_file("initialized_discriminator.pth")
        run.log_artifact(discriminator_artifact)
    return None

def train(extractor, classifier, discriminator, train_loader, val_loader, args, data_args):
    extractor = extractor.to(device=device)  # move the model parameters to CPU/GPU
    classifier = classifier.to(device=device)
    discriminator = discriminator.to(device=device)
    extractor.train()
    classifier.train()
    discriminator.train()
    example_ct = 0
    if args.num_classes==2:
        loss_fcn = nn.BCELoss()
    else:
        loss_fcn = F.cross_entropy
    optimizerC = getattr(torch.optim, args.optimizer_c)(classifier.parameters(), lr=args.lr, weight_decay=args.wd_c, \
        betas=(args.beta1, args.beta2))
    optimizerD = getattr(torch.optim, args.optimizer_d)(discriminator.parameters(), lr=args.lr_d, weight_decay=args.wd_d, \
        momentum=args.momentum)
    optimizerE = getattr(torch.optim, args.optimizer_e)(extractor.parameters(), lr=args.lr, weight_decay=args.wd_e, \
        betas=(args.beta1, args.beta2))
    earlystopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    wandb.watch(extractor, loss_fcn, log="all", log_freq=1000)
    wandb.watch(classifier, loss_fcn, log="all", log_freq=1000)
    wandb.watch(discriminator, loss_fcn, log="all", log_freq=1000)
    for e in range(args.max_epoch):
        print("this is epoch", e)
        for t, (x, y, d) in enumerate(train_loader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=dtype) if args.num_classes==2 else y.to(device=device, dtype=torch.long) 
            d = d.to(device=device, dtype=dtype) # domain label

            optimizerC.zero_grad()
            optimizerD.zero_grad()
            optimizerE.zero_grad()
            
            # train classifier and discriminator
            extractor.eval()
            classifier.train()
            discriminator.train()

            # classification loss
            feature = extractor(x)
            class_scores = torch.squeeze(classifier(feature.detach()))
            class_loss = loss_fcn(class_scores, y)
            class_loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.clip_grad_norm)
            optimizerC.step()

            # discriminator
            alpha = (args.alpha_upper - args.alpha)*e / args.max_epoch + args.alpha
            domain_scores = torch.squeeze(discriminator(feature.detach(), alpha))
            if data_args.shift_type == "domain_shift":
                weights = [data_args.shift_strength, 1-data_args.shift_strength]
                loss_weights = torch.ones(len(d)).to(device=device)
                w = torch.tensor([weights[0]]).to(device=device)
                loss_weights = loss_weights * w
                loss_weights.to(device=device)
                condition = torch.eq(d, torch.zeros(len(d)).to(device=device))
                condition = condition.to(device=device)
                w = torch.tensor([weights[1]])
                w = w.to(device=device)
                loss_weights = loss_weights.where(condition, w)
                # loss_weights = torch.FloatTensor(loss_weights).to(device=device)
                discriminator_loss_fcn = nn.BCELoss(weight=loss_weights)
            else:
                discriminator_loss_fcn = nn.BCELoss()
            domain_loss = discriminator_loss_fcn(domain_scores, d)
            domain_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.clip_grad_norm)
            optimizerD.step()

            # train encoder
            extractor.train()
            classifier.eval()
            discriminator.eval()

            feature = extractor(x)
            domain_scores = torch.squeeze(discriminator(feature, alpha))
            d = (d-1) * -1 # flip the label
            domain_loss = discriminator_loss_fcn(domain_scores, d)

            class_scores = torch.squeeze(classifier(feature))
            class_loss = loss_fcn(class_scores, y)

            e_loss = domain_loss + class_loss
            e_loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), args.clip_grad_norm)
            optimizerE.step()
            example_ct += len(x)
            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, class_loss.item()))
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, domain_loss.item()))
                train_log(class_loss, domain_loss, example_ct, e)
        class_val_loss, domain_val_loss, class_val_acc, domain_val_acc, class_val_outputs, domain_val_outputs = \
                test(extractor, classifier, discriminator, val_loader, args, data_args)
        print(class_val_outputs)
        test_log(class_val_loss, domain_val_loss, class_val_acc, domain_val_acc, example_ct, e)
        # if earlystopper.early_stop(class_val_loss) and e >= args.min_epoch: 
        #     break
    return None

def train_and_log(train_args, extractor_args, classifier_args, discriminator_args, data_args):
    with wandb.init(project="msc_project", job_type=f"dann_{data_args.shift_type}_train4", config=train_args) as run:
        config = wandb.config
        data = run.use_artifact('wav_dataset_exp001:latest')
        data_dir = data.download()
        data_dir = data_dir[2:]
        train_ds = get_dataset(data_dir, "train", shift_type=data_args.shift_type, waveshape=data_args.waveshape,\
                    shift_strength=data_args.shift_strength, shift_way=data_args.shift_way, domain_label=True)
        val_ds = get_dataset(data_dir, "val", shift_type="domain_shift", waveshape=data_args.waveshape,\
                    shift_strength=0.5, shift_way=data_args.shift_way, domain_label=True)
        data_config = vars(data_args)
        config.update(data_config)
        print("finish load data from artifact")

        train_loader = DataLoader(train_ds, batch_size=train_args.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=train_args.batch, shuffle=True)
        print("finish initialise data loader")

        extractor_artifact = run.use_artifact("extractor:latest")
        print("finish use extractor artifact")
        extractor_dir = extractor_artifact.download()
        extractor_path = os.path.join(extractor_dir[2:], "initialized_extractor.pth")
        extractor_config = extractor_artifact.metadata
        config.update(extractor_config)
        args = argparse.Namespace(**config) 
        extractor = FeatureExtractor(extractor_args)
        print("finish create extractor")
        print("extractor_dir=", extractor_dir)
        print("extractor_path=", extractor_path)
        extractor.load_state_dict(torch.load(extractor_path))

        classifier_artifact = run.use_artifact("classifier:latest")
        print("finish use classifier artifact")
        classifier_dir = classifier_artifact.download()
        classifier_path = os.path.join(classifier_dir[2:], "initialized_classifier.pth")
        classifier_config = classifier_artifact.metadata
        config.update(classifier_config)
        args = argparse.Namespace(**config) 
        classifier = Classifier(classifier_args)
        print("finish create classifier")
        print("classifier_dir=", classifier_dir)
        print("classifier_path=", classifier_path)
        classifier.load_state_dict(torch.load(classifier_path))

        discriminator_artifact = run.use_artifact("discriminator:latest")
        print("finish use discriminator artifact")
        discriminator_dir = discriminator_artifact.download()
        discriminator_path = os.path.join(discriminator_dir[2:], "initialized_discriminator.pth")
        discriminator_config = discriminator_artifact.metadata
        config.update(discriminator_config)
        args = argparse.Namespace(**config) 
        discriminator = Discriminator(discriminator_args)
        print("finish create discriminator")
        print("discriminator_dir=", discriminator_dir)
        print("discriminator_path=", discriminator_path)
        discriminator.load_state_dict(torch.load(discriminator_path))

        train(extractor, classifier, discriminator, train_loader, val_loader, train_args, data_args)

        extractor_artifact = wandb.Artifact(
            "trained-extractor", type="model",
            description="Trained extractor in DANN",
            metadata=dict(extractor_config))
        torch.save(extractor.state_dict(), "trained_extractor.pth")
        extractor_artifact.add_file("trained_extractor.pth")
        wandb.save("trained_extractor.pth")
        run.log_artifact(extractor_artifact)

        classifier_artifact = wandb.Artifact(
            "trained-classifier", type="model",
            description="Trained classifier in DANN",
            metadata=dict(classifier_config))
        torch.save(classifier.state_dict(), "trained_classifier.pth")
        classifier_artifact.add_file("trained_classifier.pth")
        wandb.save("trained_classifier.pth")
        run.log_artifact(classifier_artifact)

        discriminator_artifact = wandb.Artifact(
            "trained-discriminator", type="model",
            description="Trained discriminator in DANN",
            metadata=dict(discriminator_config))
        torch.save(discriminator.state_dict(), "trained_discriminator.pth")
        discriminator_artifact.add_file("trained_discriminator.pth")
        wandb.save("trained_discriminator.pth")
        run.log_artifact(discriminator_artifact)
    return extractor, classifier, discriminator

def test(extractor, classifier, discriminator, test_loader, args, data_args):
    extractor.eval()
    classifier.eval()
    discriminator.eval()
    class_loss = 0
    domain_loss = 0
    class_num_correct = 0
    domain_num_correct = 0
    num_samples = 0
    class_outputs = None
    domain_outputs = None
    if args.num_classes==2:
        loss_fcn = nn.BCELoss(reduction='sum')
    else:
        loss_fcn = F.cross_entropy
    with torch.no_grad():
        for t, (x, y, d) in enumerate(test_loader):
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=dtype) if args.num_classes==2 else y.to(device=device, dtype=torch.long)
            d = d.to(device=device, dtype=dtype) 
            features = extractor(x)
            class_scores = torch.squeeze(classifier(features))
            domain_scores = torch.squeeze(discriminator(features, args.alpha))
            if (class_outputs is None) and (domain_outputs is None):
                class_outputs = class_scores
                domain_outputs = domain_scores
            else:
                class_outputs = torch.cat((class_outputs, class_scores))
                domain_outputs = torch.cat((domain_outputs, domain_scores))
            if args.num_classes==2:
                class_loss += loss_fcn(class_scores, y)
                if data_args.shift_type == "domain_shift":
                    weights = [data_args.shift_strength, 1-data_args.shift_strength]
                    loss_weights = torch.ones(len(d)).to(device=device)
                    w = torch.tensor([weights[0]]).to(device=device)
                    loss_weights = loss_weights * w
                    loss_weights.to(device=device)
                    condition = torch.eq(d, torch.zeros(len(d)).to(device=device))
                    condition = condition.to(device=device)
                    w = torch.tensor([weights[1]])
                    w = w.to(device=device)
                    loss_weights = loss_weights.where(condition, w)
                    # loss_weights = torch.FloatTensor(loss_weights).to(device=device)
                    discriminator_loss_fcn = nn.BCELoss(weight=loss_weights)
                else:
                    discriminator_loss_fcn = loss_fcn
                domain_loss += discriminator_loss_fcn(domain_scores, d)
                class_preds = class_scores>0.5
                domain_preds = domain_scores>0.5
            else:
                test_loss += loss_fcn(scores, y)
                _, preds = scores.max(1)
            class_num_correct += (class_preds == y).sum()
            domain_num_correct += (domain_preds == d).sum()
            num_samples += class_preds.size(0)
            # print("class_scores", class_scores)
            # print("domain_scores", domain_scores)
            # print("class_preds:", class_preds, "y: ",y)
            # print("domain_preds:", domain_preds, "d: ",d)
            # print("class_num_correct:", class_num_correct, "num_samples:", num_samples)
            # print("domain_num_correct:", domain_num_correct, "num_samples:", num_samples)
    class_loss /= len(test_loader.dataset)
    domain_loss /= len(test_loader.dataset)
    class_acc = float(class_num_correct) / num_samples
    domain_acc = float(domain_num_correct) / num_samples
    return class_loss, domain_loss, float(class_acc), float(domain_acc), class_outputs, domain_outputs

def test_single(extractor, classifier, test_loader, args):
    extractor.eval()
    classifier.eval()
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
            features = extractor(x)
            scores = torch.squeeze(classifier(features))
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

def evaluate(extractor, classifier, discriminator, test_loader, triain_args, data_args):
    class_test_loss, domain_test_loss, class_test_acc, domain_test_acc, class_test_outputs, domain_test_outputs = \
        test(extractor, classifier, discriminator, test_loader, triain_args, data_args)
    return class_test_loss, class_test_acc, class_test_outputs

def evaluate_and_log(data_args, triain_args, model_args=None, extractor_ver="latest", classifier_ver="latest", discrim_ver="latest"):
    with wandb.init(project="msc_project", job_type=f"{data_args.shift_type}_eval4", config=model_args) as run:
        config = wandb.config
        data_config = vars(data_args)
        config.update(data_config)
        data = run.use_artifact('wav_dataset_exp001:latest')
        data_dir = data.download()
        data_dir = data_dir[2:]
        same_dist_testdata = get_dataset(data_dir, "test", shift_type=data_args.shift_type, shift_strength=data_args.shift_strength,\
            shift_way=data_args.shift_way, domain_label=True)
        anti_biased_testdata = get_dataset(data_dir, "test", shift_type=data_args.shift_type, shift_strength=data_args.shift_strength,\
            shift_way=(data_args.shift_way)[::-1], domain_label=True)
        neutral_testdata = get_dataset(data_dir, "test", shift_type="domain_shift", shift_strength=0.5,\
            shift_way=(data_args.shift_way), domain_label=True)
        sawtooth_test_data = get_dataset(data_dir, "test", waveshape="sawtooth", domain_label=False)
        triangle_test_data = get_dataset(data_dir, "test", waveshape="triangle", domain_label=False)

        batch_size = 64
        same_dist_loader = DataLoader(same_dist_testdata, shuffle=False, batch_size=batch_size)
        anti_biased_loader = DataLoader(anti_biased_testdata, shuffle=False, batch_size=batch_size)
        neutral_dist_loader = DataLoader(neutral_testdata, shuffle=False, batch_size=batch_size)
        sawtooth_test_loader = DataLoader(sawtooth_test_data, shuffle=False, batch_size=batch_size)
        triangle_test_loader = DataLoader(triangle_test_data, shuffle=False, batch_size=batch_size)

        extractor_artifact = run.use_artifact("trained-extractor:"+extractor_ver)
        extractor_dir = extractor_artifact.download()
        extractor_path = os.path.join(extractor_dir, "trained_extractor.pth")
        extractor_config = extractor_artifact.metadata
        config.update(extractor_config)
        args = argparse.Namespace(**extractor_config) 
        extractor = FeatureExtractor(args)
        extractor.load_state_dict(torch.load(extractor_path))
        extractor.to(device)

        classifier_artifact = run.use_artifact("trained-classifier:"+classifier_ver)
        classifier_dir = classifier_artifact.download()
        classifier_path = os.path.join(classifier_dir, "trained_classifier.pth")
        classifier_config = classifier_artifact.metadata
        config.update(classifier_config)
        args = argparse.Namespace(**classifier_config)
        classifier = Classifier(args)
        classifier.load_state_dict(torch.load(classifier_path))
        classifier.to(device)

        discriminator_artifact = run.use_artifact("trained-discriminator:"+discrim_ver)
        discriminator_dir = discriminator_artifact.download()
        discriminator_path = os.path.join(discriminator_dir, "trained_discriminator.pth")
        discriminator_config = discriminator_artifact.metadata
        config.update(discriminator_config)
        args = argparse.Namespace(**discriminator_config)
        discriminator = Discriminator(args)
        discriminator.load_state_dict(torch.load(discriminator_path))
        discriminator.to(device)

        same_dist_loss, same_dist_accuracy, same_dist_outputs = evaluate(extractor, classifier, discriminator, same_dist_loader, triain_args, data_args)
        print("same_dist_outputs", same_dist_outputs)
        anti_biased_loss, anti_biased_accuracy, anti_biased_outputs = evaluate(extractor, classifier, discriminator, anti_biased_loader, triain_args, data_args)
        print("anti_biased_outputs", anti_biased_outputs)
        neutral_loss, neutral_accuracy, neutral_outputs = evaluate(extractor, classifier, discriminator, neutral_dist_loader, triain_args, data_args)
        sawtooth_loss, sawtooth_accuracy, sawtooth_outputs = test_single(extractor, classifier, sawtooth_test_loader, triain_args)
        triangle_loss, triangle_accuracy, triangle_outputs = test_single(extractor, classifier, triangle_test_loader, triain_args)

        same_dist_outputs = same_dist_outputs.numpy(force=True)
        anti_biased_outputs = anti_biased_outputs.numpy(force=True)
        neutral_outputs = neutral_outputs.numpy(force=True)   
        sawtooth_outputs = sawtooth_outputs.numpy(force=True)
        triangle_outputs = triangle_outputs.numpy(force=True)

        np.savetxt('same_dist_outputs.txt', same_dist_outputs, fmt='%1.3f')
        np.savetxt('anti_biased_outputs.txt', anti_biased_outputs, fmt='%1.3f')
        np.savetxt('neutral_outputs.txt', neutral_outputs, fmt='%1.3f')
        np.savetxt('sawtooth_outputs.txt', sawtooth_outputs, fmt='%1.3f')
        np.savetxt('triangle_outputs.txt', triangle_outputs, fmt='%1.3f')
        run.summary.update({"anti_bias_loss": anti_biased_loss, "anti_bias_accuracy":anti_biased_accuracy,
                            "neutral_loss": neutral_loss, "neutral_accuracy":neutral_accuracy,
                            "sawtooth_loss": sawtooth_loss, "sawtooth_accuracy":sawtooth_accuracy, 
                            "triangle_loss": triangle_loss, "triangle_accuracy":triangle_accuracy,
                            "same_dist_loss": same_dist_loss, "same_dist_accuracy":same_dist_accuracy})
        wandb.save("same_dist_outputs.txt")
        wandb.save("anti_biased_outputs.txt")
        wandb.save("sawtooth_outputs.txt")
        wandb.save("neutral_outputs.txt")
        wandb.save("triangle_outputs.txt")
    return None

def train_log(class_loss, domain_loss, example_ct, epoch):
    class_loss = float(class_loss)
    domain_loss = float(domain_loss)
    wandb.log({"epoch": epoch, "classifier/loss": class_loss}, step=example_ct)
    wandb.log({"epoch": epoch, "discriminator/loss": domain_loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {class_loss:.3f}")
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {domain_loss:.3f}")

def test_log(class_loss, domain_loss, class_acc, domain_acc, example_ct, epoch):
    class_loss = float(class_loss)
    domain_loss = float(domain_loss)
    class_acc = float(class_acc)
    domain_acc = float(domain_acc)
    wandb.log({"epoch": epoch, "validation/class_loss": class_loss, "validation/class_acc": class_acc}, step=example_ct)
    wandb.log({"epoch": epoch, "validation/domain_loss": domain_loss, "validation/domain_acc": domain_acc}, step=example_ct)
    print(f"class_loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {class_loss:.3f}/{class_acc:.3f}")
    print(f"domain_loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {domain_loss:.3f}/{domain_acc:.3f}")

if __name__ == "__main__":
    alphas = [3]
    a_uppers = [10]
    lrs = [0.00005]
    for a in alphas:
        for l in lrs:
            for a_upper in a_uppers:
                data_args = get_data_args(shift_type="sample_selection_bias", shift_strength=1.0, shift_way=["sine", "square"],\
                        waveshape=None)
                extractor_args = get_extractor_args()
                classifier_args = get_classifier_args()
                discriminator_args = get_discriminator_args()
                training_args = get_training_args(a, a_upper, l)
                torch.cuda.manual_seed(training_args.seed) # 5, 55, 555, 5555, 55555
                build_and_log_model(extractor_args, classifier_args, discriminator_args)
                train_and_log(training_args, extractor_args, classifier_args, discriminator_args, data_args)
                evaluate_and_log(data_args, training_args)

