import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data_modelnet40 import ModelFetcher
from modules import ISAB, PMA, SAB

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

    def forward(self, X, h):
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

    def forward(self, x, h):
        X = self.enc(x)
        X, _ = X.max(1)
        X = self.dec(X)
        return X, h

class PermInvRNN(nn.Module):
    #def __init__(self, device, dim_input=3, num_outputs=1, dim_output=40, dim_hidden=256):
    def __init__(self, device, dim_input=3, num_outputs=1, dim_output=40, dim_hidden=256, dim_rnn=256, rnn_type='GRU', bptt_steps=100, rnn_dropout=0.85):
        super(PermInvRNN, self).__init__()
        self.device = device
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_rnn = dim_rnn
        self.bptt_steps = bptt_steps
        self.rnn_type = rnn_type
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()
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
            self.rnn = nn.GRU(self.dim_rnn, self.dim_rnn, num_layers=2, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.dim_rnn, self.dim_rnn, num_layers=2, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(self.dim_rnn, self.dim_rnn, num_layers=2, batch_first=True, dropout=rnn_dropout)
        else:
            raise ValueError('Illegal rnn_type.')


    def apply_rnn(self, input, states):
        if self.rnn_type == 'LSTM':
            output, (hn, cn) = self.rnn(input, states)
            return output, (hn, cn)
        else:
            output, hn = self.rnn(input, states)
            return output, hn


    def forward(self, X, hidden=None):
        X = self.enc(X)
        X, hidden = self.apply_rnn(X, hidden)
        X = X[:, -1, :]
        X = self.dec(X)
        return X, hidden
    
    def _forward(self, X):
        X = self.enc(X)
        #X = self.pre_rnn(X)
        X_no_grad = X[:,:-self.bptt_steps,:]
        X_with_grad = X[:,-self.bptt_steps:,:]
        _X, _hidden = self.rnn(X_no_grad)
        #print(_hidden.shape)
        #print(_X.shape)
        #exit()
        #for i in range(X.shape[1] - self.ptt_step):
        #    X[:,i, :].detach()
        #for i in range(_hidden.shape[1]):
        if self.rnn_type == 'LSTM':
            _hidden = tuple([v.detach() for v in _hidden])
        else:
            _hidden = _hidden.detach()
        X.detach()
        
        X, _ = self.rnn(X_with_grad, _hidden)
        X = X[:, -1, :]
        #X = self.post_rnn(X)
        X = self.dec(X)
        return X

    def regularize(self, X):
        X = self.enc(X)
        #X = self.pre_rnn(X)
        rand_prfx_seq_len = torch.randint(low=0, high=X.shape[1]-2, size=())

        #X_prefix = torch.narrow(X, dim=1, start=0, length=rand_prfx_seq_len).to(self.device)
        X_prefix = cudify(torch.narrow(X, dim=1, start=0, length=rand_prfx_seq_len))

        #X_suffix_a = torch.narrow(X, dim=1, start=rand_prfx_seq_len, length=2).to(self.device)
        X_suffix_a = cudify(torch.narrow(X, dim=1, start=rand_prfx_seq_len, length=2))
        #X_suffix_b = X_suffix_a[:, [1,0], :].to(self.device)
        X_suffix_b = cudify(X_suffix_a[:, [1,0], :])

        #X_a = torch.cat((X_prefix, X_suffix_a), dim=1).to(self.device)
        X_a = cudify(torch.cat((X_prefix, X_suffix_a), dim=1))
        #X_b = torch.cat((X_prefix, X_suffix_b), dim=1).to(self.device)
        X_b = cudify(torch.cat((X_prefix, X_suffix_b), dim=1))
        #X_a_no_grad = X_a[:,:-self.bptt_steps,:]
        #X_b_no_grad = X_b[:,:-self.bptt_steps,:]
        #X_a_with_grad = X_a[:,-self.bptt_steps:,:]
        #X_b_with_grad = X_b[:,-self.bptt_steps:,:]
        #_output_a, _hidden_a = self.rnn(X_a_no_grad) 
        #_output_b, _hidden_b = self.rnn(X_b_no_grad) 
        _output_a, _hidden_a = self.rnn(X_a)
        _output_b, _hidden_b = self.rnn(X_b)
        
        # if self.rnn_type == 'LSTM':
        #     _hidden_a = tuple([v.detach() for v in _hidden_a])
        #     _hidden_b = tuple([v.detach() for v in _hidden_b])
        # else:
        #     _hidden_a = _hidden_a.detach()
        #     _hidden_b = _hidden_b.detach()
        
        # _output_a.detach()
        # _output_b.detach()
        # output_a, hidden_a = self.rnn(X_a, _hidden_a)
        # output_b, hidden_b = self.rnn(X_b, _hidden_b)
        
        #output_a[:,:-50,:].detach()
        #output_b[:,:-50,:].detach()
        output_a = output_a[:, -1, :]
        output_b = output_b[:, -1, :]

        #hidden_a = hidden_a[0]
        #hidden_b = hidden_b[0]

        #diff = (hidden_a - hidden_b).pow(2).sum() / X.shape[0]
        diff = (output_a - output_b).pow(2).sum() / X.shape[0]
        return diff

parser = argparse.ArgumentParser()
parser.add_argument("--num_pts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--dim_rnn", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=2000)
parser.add_argument("--model", type=str, default="reg")
parser.add_argument("--rnn_type", type=str, default="GRU")


args = parser.parse_args()
#args.exp_name = f"N{args.num_pts}_d{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}bs{args.batch_size}"
from time import gmtime, strftime
strftime("%Y-%m-%d_%H:%M", gmtime())
#args.exp_name = f"pts-{args.num_pts}_dim-{args.dim}_dimrnn-{args.dim_rnn}_lr-{args.learning_rate}_bs-{args.batch_size}_rnn_type-{args.rnn_type}_epochs-{args.train_epochs}"
model_name = f"model-{args.model}"
if args.model == 'reg':
    model_name += f"_rnn_type-{args.rnn_type}_dimrnn-{args.dim_rnn}"
args.exp_name = f"pts-{args.num_pts}_dim-{args.dim}_lr-{args.learning_rate}_bs-{args.batch_size}_{model_name}_epochs-{args.train_epochs}"

log_dir_base = "result/" + args.exp_name
if not os.path.exists(log_dir_base):
    os.makedirs(log_dir_base)

curr_exp = strftime("%Y-%m-%d_%H:%M", gmtime())

log_dir = os.path.join(os.path.join(log_dir_base, curr_exp))
log_file = os.path.join(log_dir, 'log.txt')

#model_path = log_dir + "/model"
#writer = SummaryWriter(log_dir)
log_dict = {
    'train_losses': [],
    'train_acc': [],
    'val_losses': [],
    'val_acc': [],
    'test_loss': [],
    'test_acc': [],
}

generator = ModelFetcher(
#    "../dataset/ModelNet40_cloud.h5",
    "/specific/netapp5_2/gamir/edocohen/TCRNN/data/PointClouds/ModelNet40_cloud_from_edo.h5",
    #"ModelNet40_cloud_from_edo.h5",
    args.batch_size,
    down_sample=int(10000 / args.num_pts),
    do_standardize=True,
    do_augmentation=(args.num_pts == 5000),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = SetTransformer(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc)
#model = DeepSet(dim_hidden=args.dim)
if args.model == 'reg':
    model = PermInvRNN(device=device, dim_hidden=args.dim, dim_rnn=args.dim_rnn, rnn_type=args.rnn_type)
elif args.model == 'deepset':
    model = DeepSet(dim_hidden=args.dim)
elif args.model == 'set':
    model = SetTransformer(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc)
else:
    raise ValueError('invalid model.')

def save_model(model, fname):
    torch.save(model, fname)

def load_model(fname):
    return torch.load(fname)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
#model = nn.DataParallel(model)
model = model.to(device)
best_val_epoch = 0
best_val_acc = 0.0
for epoch in range(args.train_epochs):
    losses, total, correct = [], 0, 0
    for imgs, _, lbls in generator.train_data():
        states = None
        imgs = torch.Tensor(imgs).to(device)
        lbls = torch.Tensor(lbls).long().to(device)
        for i, imgs_ in enumerate(imgs.split(model.bptt_steps, dim=1)):
            total_loss = 0.0
            optimizer.zero_grad()
            model.train()
            if states is not None:
                # LSTM returns a tuple.
                if args.rnn_type == 'LSTM':
                    states[0] = states[0].detach()
                    states[1] = states[1].detach()
                else:    
                    states = states.detach()
            
            preds, states = model(imgs_, states)
            reg_loss = model.regularize(imgs_) if args.model == 'reg' else 0.0
            
            loss = criterion(preds, lbls)
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()

        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls).sum().item()

    avg_loss, avg_acc = np.mean(losses), correct / total
    log_dict['train_losses'].append(avg_loss)
    log_dict['train_acc'].append(avg_acc)
    #writer.add_scalar("train_loss", avg_loss, epoch)
    #writer.add_scalar("train_acc", avg_acc, epoch)
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

    if epoch % 5 == 0:
        model.eval()
        losses, total, correct = [], 0, 0
        #for imgs, _, lbls in generator.test_data():
        for imgs, _, lbls in generator.val_data():
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            if args.model == 'reg':
                # reg_loss = model.regularize(imgs)
                preds, states = model(imgs, states)
            else:
                preds = model(imgs)
            
            loss = criterion(preds, lbls)
            losses.append(loss.item())
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()
        avg_loss, avg_acc = np.mean(losses), correct / total
        log_dict['val_losses'].append(avg_loss)
        log_dict['val_acc'].append(avg_acc)
        #writer.add_scalar("val_loss", avg_loss, epoch)
        #writer.add_scalar("val_acc", avg_acc, epoch)
        print(f"Epoch {epoch}: val loss {avg_loss:.3f} val acc {avg_acc:.3f}")
        if avg_acc > best_val_acc:
            print(f"Saving best model. Best val accuracy: {avg_acc:.3f}. Epoch {epoch}")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            best_val_acc = avg_acc
            best_val_epoch = epoch
            #save_model(model, os.path.join(log_dir, curr_exp + '_best_val.pth'))
            #save_model(model, os.path.join(log_dir, 'best_val.pth')
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_val.pth'))
            log_dict['best_epoch'] = epoch


print("Loading best model (epoch: {})".format(log_dict['best_epoch']))
#model = load_model(os.path.join(log_dir, 'best_val.pth'))
model.load_state_dict(torch.load(os.path.join(log_dir, 'best_val.pth')))
model.eval()
losses, total, correct = [], 0, 0
#for imgs, _, lbls in generator.test_data():
for imgs, _, lbls in generator.test_data():
    imgs = torch.Tensor(imgs).cuda()
    lbls = torch.Tensor(lbls).long().cuda()
    preds = model(imgs)
    loss = criterion(preds, lbls)

    losses.append(loss.item())
    total += lbls.shape[0]
    correct += (preds.argmax(dim=1) == lbls).sum().item()
avg_loss, avg_acc = np.mean(losses), correct / total
log_dict['test_loss'].append(avg_loss)
log_dict['test_acc'].append(avg_acc) 
#save_model(model, os.path.join(log_dir, 'final.pth'))
#writer.add_scalar("test_loss", avg_loss)
#writer.add_scalar("test_acc", avg_acc)
print(f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")
with open(log_file, 'w') as f:
    json.dump(log_dict, f)
