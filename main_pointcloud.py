import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data_modelnet40 import ModelFetcher
from modules import ISAB, PMA, SAB

from time import gmtime, strftime
import json

cudify = lambda x: x.cuda() if torch.cuda.is_available() else x

class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X, h=None):
        return self.dec(self.enc(X)).squeeze(), h

class PermEqui1_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        x = self.Gamma(x-xm)
        return x

class DeepSet(nn.Module):
    def __init__(self, dim_input=3, num_outputs=1, dim_output=40, dim_hidden=256):
        super(DeepSet, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                PermEqui1_max(self.dim_input, self.dim_hidden),
                nn.Tanh(),
                PermEqui1_max(self.dim_hidden, self.dim_hidden),
                nn.Tanh(),
                PermEqui1_max(self.dim_hidden, self.dim_hidden),
                nn.Tanh())
        self.dec = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.Tanh(),
                nn.Dropout(p=0.5),
                nn.Linear(self.dim_hidden, self.dim_output))

    def forward(self, x, h=None):
        X = self.enc(x)
        X, _ = X.max(1)
        X = self.dec(X)
        return X, h

class PermInvRNN(nn.Module):
    #def __init__(self, device, dim_input=3, num_outputs=1, dim_output=40, dim_hidden=256):
    def __init__(self, dim_input=3, num_outputs=1, dim_output=40, dim_hidden=256, dim_rnn=256, num_rnn_layers=2, rnn_type='GRU', sparse_rnn=False, activation='tanh', rnn_dropout=0.85):
        super(PermInvRNN, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_rnn = dim_rnn
        self.num_rnn_layers = num_rnn_layers
        self.rnn_type = rnn_type        
        self.sparse_rnn = sparse_rnn
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('Invalid activation.')

        self.enc = nn.Sequential(
                PermEqui1_max(self.dim_input, self.dim_hidden),
                self.activation,
                PermEqui1_max(self.dim_hidden, self.dim_hidden),
                self.activation,
                PermEqui1_max(self.dim_hidden, self.dim_rnn),
                self.activation)
        self.dec = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.dim_rnn, self.dim_hidden),
                self.activation,
                nn.Dropout(p=0.5),
                nn.Linear(self.dim_hidden, self.dim_output))

        if self.rnn_type == 'GRU':
            rnn_obj = nn.GRU
        elif self.rnn_type == 'LSTM':
            rnn_obj = nn.LSTM
        elif self.rnn_type == 'RNN':
            rnn_obj = nn.RNN
        else:
            raise ValueError('Illegal rnn_type.')

        if self.sparse_rnn:
            print('sparse rnn')
            self.rnn = [rnn_obj(1, 1, num_layers=self.num_rnn_layers, batch_first=True, dropout=rnn_dropout) for i in range(self.dim_rnn)]
        else:
            print('non sparse rnn')
            self.rnn = rnn_obj(self.dim_rnn, self.dim_rnn, num_layers=self.num_rnn_layers,
                                                        batch_first=True, dropout=rnn_dropout)

        self.rnn_dropout = nn.Dropout(p=rnn_dropout)



    def apply_rnn(self, input, states=None):
        if self.sparse_rnn:
            inputs = torch.split(input, 1, dim=-1)
            if states is not None:
                states = torch.split(states, 1, dim=-1)
            else:
                states = [None for i in range(len(inputs))]

            rnn_output_arr = [self.rnn[i](curr_input, curr_state) for i, (curr_input, curr_state) in enumerate(zip(inputs, states))]
            output_arr = [_[0] for _ in rnn_output_arr]
            next_state_arr = [_[1] for _ in rnn_output_arr]
            output = torch.cat(output_arr, dim=-1)
            next_state = torch.cat(next_state_arr, dim=-1)
            return output, next_state
        else:
            if self.rnn_type == 'LSTM':
                output, (hn, cn) = self.rnn(input, states)
                # return output, (hn, cn)
                next_state = (hn, cn)
            else:
                output, c = self.rnn(input, states)
                next_state = hn
            
            return self.rnn_dropout(output), next_state


    def forward(self, X, hidden=None):
        X = self.enc(X)
        # X, hidden = self.apply_rnn(X)
        X, hidden = self.apply_rnn(X, hidden)
        X = X[:, -1, :]
        X = self.dec(X)
        return X, hidden

    def regularize(self, X):
        X = self.enc(X)
        rand_prfx_seq_len = torch.randint(low=0, high=X.shape[1]-2, size=())

        X_prefix = cudify(torch.narrow(X, dim=1, start=0, length=rand_prfx_seq_len))
        X_suffix_a = cudify(torch.narrow(X, dim=1, start=rand_prfx_seq_len, length=2))
        X_suffix_b = cudify(X_suffix_a[:, [1,0], :])

        X_a = cudify(torch.cat((X_prefix, X_suffix_a), dim=1))
        X_b = cudify(torch.cat((X_prefix, X_suffix_b), dim=1))
        output_a, hidden_a = self.apply_rnn(X_a)
        output_b, hidden_b = self.apply_rnn(X_b)
        # output_a, hidden_a = self.rnn(X_a)
        # output_b, hidden_b = self.rnn(X_b)

        output_a = output_a[:, -1, :]
        output_b = output_b[:, -1, :]

        #hidden_a = hidden_a[0]
        #hidden_b = hidden_b[0]

        #diff = (hidden_a - hidden_b).pow(2).sum() / X.shape[0]
        diff = (output_a - output_b).pow(2).sum() / X.shape[0]
        return diff

def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm

def _detach(var):
    if type(var) is tuple:
        var[0] = var[0].detach()
        var[1] = var[1].detach()
    else:
        var = var.detach()
    return var

def load_model(args):
        if args.model == 'reg':
            model = PermInvRNN(
                        dim_hidden=args.dim,
                        dim_rnn=args.dim_rnn, 
                        num_rnn_layers=args.num_rnn_layers,
                        rnn_type=args.rnn_type,
                        sparse_rnn=(args.rnn_structure == 'sparse'),
                        activation=args.rnn_activation,
                        rnn_dropout=args.rnn_dropout
            )
        elif args.model == 'deepset':
            model = DeepSet(dim_hidden=args.dim)
        elif args.model == 'set':
            model = SetTransformer(
                dim_hidden=args.dim,
                num_heads=args.n_heads,
                num_inds=args.n_anc
            )
        else:
            raise ValueError('invalid model.')
        return model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pts", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--reg_coef", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bptt_steps", type=int, default=100)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dim_rnn", type=int, default=256)
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_anc", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=2000)
    parser.add_argument("--model", type=str, default="reg")
    parser.add_argument("--rnn_type", type=str, default="GRU")
    parser.add_argument("--rnn_structure", type=str, default="regular")
    # parser.add_argument("--sparse_rnn", type=str, default=False)
    parser.add_argument("--rnn_activation", type=str, default="relu")
    parser.add_argument("--rnn_dropout", type=float, default=0.85)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--log_dir", type=str, default=None)

    parser.set_defaults(sparse_rnn=False)
    args = parser.parse_args()
    return args

def train(args, generator):
    model_name = f"model-{args.model}"
    if args.model == 'reg':
        model_name += (
            f"_rnn_type-{args.rnn_type}"
            f"_rnn_sttructure-{args.rnn_structure}"
            f"_dimrnn-{args.dim_rnn}"
            f"_rnnlayers-{args.num_rnn_layers}"
            f"_bpttsteps-{args.bptt_steps}"
            f"_rnn_dropout-{args.rnn_dropout}"
            f"_rnn_activation-{args.rnn_activation}"
            f"_reg_coef-{args.reg_coef}"
        )
    else:
        # truncated backprop is relevant only to the RNN...
        args.bptt_steps = args.num_pts

    args.exp_name = (
            f"pts-{args.num_pts}"
            f"_dim-{args.dim}"
            f"_lr-{args.learning_rate}"
            f"_opt-{args.optimizer}"
            f"_weight_decay-{args.weight_decay}"
            f"_eps-{args.eps}"
            f"_bs-{args.batch_size}"
            f"_{model_name}"
            f"_epochs-{args.train_epochs}"
        )

    log_dir_base = "result/" + args.exp_name
    if not os.path.exists(log_dir_base):
        os.makedirs(log_dir_base)

    curr_exp = strftime("%Y-%m-%d_%H:%M", gmtime())

    log_dir = os.path.join(os.path.join(log_dir_base, curr_exp))
    config_file = os.path.join(log_dir, 'config.txt')

    log_dict = {
        'log_dir': log_dir,
        'train_losses': [],
        'train_acc': [],
        'val_losses': [],
        'val_acc': [],
        # 'test_loss': [],
        # 'test_acc': [],
    }
        
    model = load_model(args)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                            lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.eps)
    elif args.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(),
                            lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.eps)
    elif args.optimizer == 'RMSProp':
        optimizer = torch.optim.RMSProp(model.parameters(),
                            lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.eps)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(),
                            lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.eps)
    else:
        raise ValueError('Illegal optimizer value')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                milestones=list(range(400, args.train_epochs, 400)), gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    model = cudify(model)
    best_val_acc_epoch = 0
    best_val_acc = 0.0
    best_val_loss_epoch = 0
    best_val_loss = np.inf
    for epoch in range(args.train_epochs):
        losses, total, correct = [], 0, 0
        for imgs, _, lbls in generator.train_data():
            states = None
            imgs = cudify(torch.Tensor(imgs))
            lbls = cudify(torch.Tensor(lbls).long())
            for i, imgs_ in enumerate(imgs.split(args.bptt_steps, dim=1)):
                total_loss = 0.0
                optimizer.zero_grad()
                model.train()
                if states is not None:
                    states = _detach(states)
                
                preds, states = model(imgs_, states)
                loss = criterion(preds, lbls)
                loss.backward()
                
                if args.model == 'reg':
                    reg_loss = model.regularize(imgs_)
                    reg_loss.backward()
                
                # total_loss = loss + args.reg_coef * reg_loss
                
                # total_loss.backward()
                clip_grad(model, 5)
                optimizer.step()

            losses.append(loss.item())
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()

        avg_loss, avg_acc = np.mean(losses), correct / total
        log_dict['train_losses'].append(avg_loss)
        log_dict['train_acc'].append(avg_acc)
        print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")
        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            losses, total, correct = [], 0, 0
            for imgs, _, lbls in generator.val_data():
                imgs = cudify(torch.Tensor(imgs))
                lbls = cudify(torch.Tensor(lbls).long())
                preds, states = model(imgs)
                
                loss = criterion(preds, lbls)
                losses.append(loss.item())
                total += lbls.shape[0]
                correct += (preds.argmax(dim=1) == lbls).sum().item()
            avg_loss, avg_acc = np.mean(losses), correct / total
            log_dict['val_losses'].append(avg_loss)
            log_dict['val_acc'].append(avg_acc)
            print(f"Epoch {epoch}: val loss {avg_loss:.3f} val acc {avg_acc:.3f}")
            if avg_acc > best_val_acc or avg_loss < best_val_loss:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                    with open(config_file, 'w') as f:
                        json.dump(args.__dict__, f, indent=2)

                if avg_acc > best_val_acc:
                    print(f"Saving best model. Best val accuracy: {avg_acc:.3f}. Epoch {epoch}")
                    best_val_acc = avg_acc
                    best_val_acc_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(log_dir, 'best_val_acc.pth'))
                    log_dict['best_val_acc_epoch'] = epoch
                
                if avg_loss < best_val_loss:
                    print(f"Saving best model. Best val loss: {avg_loss:.3f}. Epoch {epoch}")
                    best_val_loss = avg_loss
                    best_val_loss_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(log_dir, 'best_val_loss.pth'))
                    log_dict['best_val_loss_epoch'] = epoch
                
                #save_model(model, os.path.join(log_dir, curr_exp + '_best_val.pth'))
                #save_model(model, os.path.join(log_dir, 'best_val.pth')
                
    with open(os.path.join(log_dir, 'log_file.txt'), 'w') as f:
        json.dump(log_dict, f)
    return log_dict['log_dir']


def eval(log_dir, args, generator, model_name):
    print("Loading model from {}".format(log_dir))
    with open(os.path.join(log_dir, 'config.txt'), 'r') as f:
        args.__dict__ = json.load(f)

    model = load_model(args)
    model.load_state_dict(torch.load(os.path.join(log_dir, model_name)))
    # model.load_state_dict(torch.load(os.path.join(log_dir, 'best_val.pth')))
    criterion = nn.CrossEntropyLoss()

    model = cudify(model)
    model.eval()
    losses, total, correct = [], 0, 0
    #for imgs, _, lbls in generator.test_data():
    for imgs, _, lbls in generator.test_data():
        imgs = cudify(torch.Tensor(imgs))
        lbls = cudify(torch.Tensor(lbls).long())
        preds, states = model(imgs)
        loss = criterion(preds, lbls)
        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls).sum().item()
    avg_loss, avg_acc = np.mean(losses), correct / total
    print(f"Test loss {avg_loss:.3f} test acc {avg_acc:.3f}")
    log_dict = {
        'batch_size': generator.batch_size,
        'pts': int(10000 / generator.down_sample),
        'loss': avg_loss,
        'acc': avg_acc,
    }
    return log_dict

if __name__=='__main__':
    args = get_parser()
    generator = ModelFetcher(
        # "/specific/netapp5_2/gamir/edocohen/TCRNN/data/PointClouds/ModelNet40_cloud_from_edo.h5",
        "ModelNet40_cloud_from_edo.h5",
        args.batch_size,
        down_sample=int(10000 / args.num_pts),
        do_standardize=True,
        do_augmentation=(args.num_pts == 5000),
    )
    if args.train_epochs > 0:
        log_dir = train(args, generator)
    else:
        log_dir = args.log_dir
        if log_dir is None:
            raise ValueError('Must provide a log_dir if not training.')

    log_dict = {}
    _log_dict = eval(log_dir, args, generator, 'best_val_acc.pth')
    log_dict['best_val_acc.pth'] = deepcopy(_log_dict)
    _log_dict = eval(log_dir, args, generator, 'best_val_loss.pth')
    log_dict['best_val_loss.pth'] = deepcopy(_log_dict)
    curr_time = strftime("%Y-%m-%d_%H:%M", gmtime())
    print(log_dict)
    with open(os.path.join(log_dir, '{}_eval.txt'.format(curr_time)), 'w') as f:
        json.dump(log_dict, f)

