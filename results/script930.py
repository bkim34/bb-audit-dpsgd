"""Auditing DP-SGD in black-box setting"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import argparse
from opacus.accountants.utils import get_noise_multiplier
import copy
from torch.utils.data import TensorDataset, DataLoader
import dill

from models import Models
from utils.data import load_data
from utils.dpsgd import clip_and_accum_grads
from utils.audit import compute_eps_lower_from_mia
from utils.clipbkd import craft_clipbkd, choose_worstcase_label

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from copy import deepcopy

from models import Models
from utils.data import load_data
from legacy.audit_model import test_model

import sys
import tensorflow as tf1
import numpy as np
import csv
import os
import random as random
from types import SimpleNamespace
import csv
import json

def _pick_device():
    # Prefer Apple GPU (MPS) on Mac, else CUDA if present, else CPU
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

device = _pick_device()

###############################################
# Pre-train on half of MNIST
###############################################

# hyper-parameters
data_name = 'mnist'
lr = 0.01
n_epochs = 5
batch_size = 32

# reproducibility
np.random.seed(0)
torch.manual_seed(0)

# load full dataset
X, y, out_dim = load_data(data_name, None, device=device, split='train')
X_test, y_test, _ = load_data(data_name, None, device=device, split='test')

len(X)

# use only first half of dataset for pre-training
X_train, y_train = X[:len(X)//2], y[:len(X)//2]

len(X_train)

# define model
model = Models['cnn'](X_train.shape, out_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr)

# train model
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)

pbar = tqdm(range(n_epochs))
losses = []
save_model_epochs = [1, 2, 3, 4]
saved_models = []
for curr_epoch in pbar:
    for curr_X, curr_y in train_loader:
        optimizer.zero_grad()

        output = model(curr_X)
        loss = criterion(output, curr_y)
        loss.backward()

        optimizer.step()

        losses.append(loss.cpu().item())
        pbar.set_postfix({'loss': losses[-1]})
    
    if curr_epoch in save_model_epochs:
        saved_models.append(deepcopy(model)) 

model.load_state_dict(torch.load('pretrained_models/cnn_mnist_half.pt'))

# test accuracy
test_acc = test_model(model, X_test, y_test) * 100
print(f'Test accuracy (%): {test_acc:.3f}')

# save model
torch.save(model.cpu().state_dict(), f'pretrained_models/cnn_mnist_half.pt')
os.makedirs('pretrained_models/cnn_mnist_half_epochs', exist_ok=True)
for i, (save_model_epoch, model) in enumerate(zip(save_model_epochs, saved_models)):
    torch.save(model.cpu().state_dict(), f'pretrained_models/cnn_mnist_half_epochs/{save_model_epoch}epochs.pt')
# save remaining half to ensure no overlap
folder = f'data/{data_name}_finetune_half/'
os.makedirs(folder, exist_ok=True)

X_finetune, y_finetune = X[len(X)//2:], y[len(y)//2:]

np.save(f'{folder}/X_train.npy', X_finetune.cpu().numpy())
np.save(f'{folder}/y_train.npy', y_finetune.cpu().numpy())
np.save(f'{folder}/X_test.npy', X_test.cpu().numpy())
np.save(f'{folder}/y_test.npy', y_test.cpu().numpy())

print("PRETRAINING COMPLETE")

###############################################
# Auditing Functions
###############################################

def xavier_init_model(model):
    """Initialize model using Xavier initialization"""
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

def train_model(model_name, X, y, epsilon, delta, max_grad_norm, n_epochs, lr, device='cpu', init_model=None, block_size=1024, out_dim=10):
    """Train model w/ DP-SGD (no sub-sampling + gradients are summed instead of averaged)"""
    # initialize model, loss function, and optimizer
    if init_model is None:
        model = Models[model_name](X.shape, out_dim=out_dim).to(device)
        xavier_init_model(model)
    else:
        model = copy.deepcopy(init_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # set noise level
    if epsilon is not None:
        # no subsampling, i.e., sample rate = 1
        noise_multiplier = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta, sample_rate=1,
            epochs=n_epochs, accountant='prv')
    else:
        noise_multiplier = 0
    
    # train model for n_epochs
    grad_norms = []
    for epoch in tqdm(range(n_epochs), leave=False):
        optimizer.zero_grad()

        accum_grad, curr_grad_norms = clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm, block_size=block_size)
        if epoch == 0:
            # save per-sample gradient norms from first epoch
            grad_norms.append(curr_grad_norms)

        # accumulate per-sample gradients and add noise
        with torch.no_grad():
            for name, param in model.named_parameters():
                curr_grad = accum_grad[name]

                if noise_multiplier > 0 and max_grad_norm is not None:
                    # add noise
                    curr_grad = curr_grad + noise_multiplier * max_grad_norm * torch.randn_like(curr_grad)
                
                # update gradient of parameter
                param.grad = curr_grad
        
        # update parameter
        optimizer.step()
    
    return model, grad_norms

def test_model(model, X, y, batch_size=128):
    """Test trained model on test set"""
    test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    model.eval()
    acc = 0
    with torch.no_grad():
        for curr_X, curr_y in test_loader:
            curr_y_hat = torch.argmax(model(curr_X), dim=1)
            acc += torch.sum(curr_y_hat == curr_y).cpu().item()
    model.train()
    
    return acc / len(y)

def save_checkpoint(out_folder, outputs, losses, all_grad_norms, train_set_accs, test_set_accs, fit_world_only, save_grad_norms):
    """Save checkpoint"""
    # create folder if not exists
    os.makedirs(out_folder, exist_ok=True)

    # save random state
    random_state = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state()
    }
    dill.dump(random_state, open(f'{out_folder}/random_state.dill', 'wb'))

    # save intermediate values
    if fit_world_only:
        np.save(f'{out_folder}/outputs_{fit_world_only}.npy', outputs[fit_world_only])
        np.save(f'{out_folder}/losses_{fit_world_only}.npy', losses[fit_world_only])
        if save_grad_norms:
            np.save(f'{out_folder}/all_grad_norms_{fit_world_only}.npy', all_grad_norms[fit_world_only])

        if fit_world_only == 'out':
            np.save(f'{out_folder}/train_set_accs.npy', train_set_accs)
            np.save(f'{out_folder}/test_set_accs.npy', test_set_accs)
    else:
        np.save(f'{out_folder}/outputs_in.npy', outputs['in'])
        np.save(f'{out_folder}/outputs_out.npy', outputs['out'])
        np.save(f'{out_folder}/train_set_accs.npy', train_set_accs)
        np.save(f'{out_folder}/test_set_accs.npy', test_set_accs)
        np.save(f'{out_folder}/losses_in.npy', losses['in'])
        np.save(f'{out_folder}/losses_out.npy', losses['out'])
        if save_grad_norms:
            np.save(f'{out_folder}/all_grad_norms_in.npy', all_grad_norms['in'])
            np.save(f'{out_folder}/all_grad_norms_out.npy', all_grad_norms['out'])

def resume_checkpoint(out_folder, save_grad_norms, fit_world_only, resume):
    """Load checkpoint if resume is set to True and previous checkpoint exists, else create new empty checkpoint"""
    outputs = {'out': [], 'in': []}
    losses = {'out': [], 'in': []}
    all_grad_norms = { 'out': [], 'in': [] }
    train_set_accs = []
    test_set_accs = []

    if os.path.exists(out_folder) and resume:
        # if folder exists and resume is set to true load previous values
        random_state = dill.load(open(f'{out_folder}/random_state.dill', 'rb'))
        np.random.set_state(random_state['np'])
        torch.random.set_rng_state(random_state['torch'])

        if fit_world_only:
            outputs[fit_world_only] = np.load(f'{out_folder}/outputs_{fit_world_only}.npy').tolist()
            losses[fit_world_only] = np.load(f'{out_folder}/losses_{fit_world_only}.npy').tolist()
            if save_grad_norms:
                all_grad_norms[fit_world_only] = np.load(f'{out_folder}/all_grad_norms_{fit_world_only}.npy').tolist()

            if fit_world_only == 'out':
                train_set_accs = np.load(f'{out_folder}/train_set_accs.npy').tolist()
                test_set_accs = np.load(f'{out_folder}/test_set_accs.npy').tolist()
        else:
            outputs['in'] = np.load(f'{out_folder}/outputs_in.npy').tolist()
            outputs['out'] = np.load(f'{out_folder}/outputs_out.npy').tolist()
            train_set_accs = np.load(f'{out_folder}/train_set_accs.npy').tolist()
            test_set_accs = np.load(f'{out_folder}/test_set_accs.npy').tolist()
            losses['in'] = np.load(f'{out_folder}/losses_in.npy').tolist()
            losses['out'] = np.load(f'{out_folder}/losses_out.npy').tolist()
            if save_grad_norms:
                all_grad_norms['in'] = np.load(f'{out_folder}/all_grad_norms_in.npy').tolist()
                all_grad_norms['out'] = np.load(f'{out_folder}/all_grad_norms_out.npy').tolist()
    else:
        # create folder and dump initial values in
        os.makedirs(out_folder, exist_ok=True)
        save_checkpoint(out_folder, outputs, losses, all_grad_norms, train_set_accs, test_set_accs, args.fit_world_only, args.save_grad_norms)
    
    return outputs, losses, all_grad_norms, train_set_accs, test_set_accs


###############################################
# Renyi Estimate
###############################################

def renyi_estimate(alpha, Lrate, epochs, hidden_layers, mb, P, Q, N):
    tf1.compat.v1.disable_eager_execution()  
    tf = tf1.compat.v1 
    np.random.seed(0); tf.set_random_seed(0); random.seed(0)
    #alpha     = 2      
    n         = 1       # feature dimension
    Lrate     = Lrate
    #epochs = 1000
    mb_size = mb
    #number of nodes in each hidden layer (can have more than one hidden layer)
    #hidden_layers = [256] 

    #save estimate every SF iterations
    SF = 250
    #samples for estimating Df
    #N=1000

    Q_pool = np.asarray(Q,  dtype=np.float32).reshape(-1, 1)
    P_pool = np.asarray(P, dtype=np.float32).reshape(-1, 1)

    n = P_pool.shape[1]

    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    #construct variables for the neural networks
    def initialize_W(layers):
        W_init=[]
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W_init.append(xavier_init(size=[layers[l], layers[l+1]]))
        return W_init

    def initialize_NN(layers,W_init):
        NN_W = []
        NN_b = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = tf.Variable(W_init[l])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            NN_W.append(W)
            NN_b.append(b)
        return NN_W, NN_b

    #variable for Q
    X = tf.placeholder(tf.float32, shape=[None, n])
    #variable for P
    Z = tf.placeholder(tf.float32, shape=[None, n])

    layers=[n]+hidden_layers +[1]

    W_init=initialize_W(layers)
    D_W, D_b = initialize_NN(layers,W_init)
    theta_D = [D_W, D_b]

    def discriminator(x):
        num_layers = len(D_W) + 1
        h = x
        for l in range(0,num_layers-2):
            W = D_W[l]
            b = D_b[l]
            h = tf.nn.relu(tf.add(tf.matmul(h, W), b))
        W = D_W[-1]
        b = D_b[-1]
        out =  tf.matmul(h, W) + b

        return out      

    def sample_P(N_samp):
        idx = np.random.randint(0, P_pool.shape[0], size=N_samp)
        return P_pool[idx]  # (N_samp, n)

    def sample_Q(N_samp):
        idx = np.random.randint(0, Q_pool.shape[0], size=N_samp)
        return Q_pool[idx]

    P_data=discriminator(Z)
    Q_data=discriminator(X)

    P_max=tf.reduce_max((alpha-1.0)*P_data)
    Q_max=tf.reduce_max(alpha*Q_data)
    alpha_scaling = 1
    #DV renyi objective
    objective=alpha_scaling/(alpha-1.0)*tf.math.log(tf.reduce_mean(tf.math.exp(((alpha-1.0)*P_data-P_max)/alpha_scaling)))+1.0/(alpha-1.0)*P_max-1.0/alpha*Q_max-alpha_scaling/alpha*tf.math.log(tf.reduce_mean(tf.math.exp((alpha*Q_data-Q_max)/alpha_scaling)))

    #AdamOptimizer
    solver = tf.train.AdamOptimizer(learning_rate=Lrate).minimize(-objective, var_list=theta_D)

    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    divergence_array=np.zeros(epochs//SF+1)

    #use N samples for estimation of Df
    Q_plot_samples=sample_Q(N)
    P_plot_samples=sample_P(N)
    j=0
    for it in range(epochs+1):
        if it>0:
                X_samples=sample_Q(mb_size)
                Z_samples=sample_P(mb_size) 

                sess.run(solver, feed_dict={X: X_samples, Z: Z_samples})
        if it % SF == 0:                 
            X_samples=Q_plot_samples
            Z_samples=P_plot_samples
            
            divergence_array[j]=sess.run( objective, feed_dict={X: X_samples, Z: Z_samples})
            j=j+1

    # print(f"Final DV Rényi estimate D_{alpha}(P||Q): {divergence_array[-1]:.6f}")
    # print(divergence_array)
    return divergence_array[-1]


###############################################
# Train Models
###############################################

def train_models(data_name="mnist",          # mnist | cifar10 | cifar100
    model_name="lr",            # try 'cnn' if you crafted a CNN init
    n_reps=200,                 # total models (split across IN/OUT worlds)
    n_df=0,                     # 0 => full dataset
    n_epochs=100,
    lr=0.01,
    max_grad_norm=1.0,
    epsilon=6.0,                # target ε (Opacus/PRV)
    delta=1e-2,
    target_type="clipbkd",      # 'blank' | 'clipbkd' | path to a target sample
    seed=0,
    out="exp_data",
    device=_pick_device(),      # "mps" on Apple GPU, else "cuda:0" or "cpu"
    fixed_init=None,     # path => worst-case; '' => fixed random; None => average-case Xavier
    block_size=1000,
    resume=True,                # skip work if results are already present
    fit_world_only=None,        # 'in' or 'out' to run a single world; None = both then combine
    save_grad_norms=False,
    alpha=0.05):                # significance for empirical ε estimate)
    # reproducibility
    args = SimpleNamespace(
        data_name=data_name,          # mnist | cifar10 | cifar100
        model_name=model_name,            # try 'cnn' if you crafted a CNN init
        n_reps=n_reps,                 # total models (split across IN/OUT worlds)
        n_df=n_df,                     # 0 => full dataset
        n_epochs=n_epochs,
        lr=lr,
        max_grad_norm=max_grad_norm,
        epsilon=epsilon,                # target ε (Opacus/PRV)
        delta=delta,
        target_type=target_type,      # 'blank' | 'clipbkd' | path to a target sample
        seed=seed,
        out=out,
        device=device,      # "mps" on Apple GPU, else "cuda:0" or "cpu"
        fixed_init=fixed_init,     # path => worst-case; '' => fixed random; None => average-case Xavier
        block_size=block_size,
        resume=resume,                # skip work if results are already present
        fit_world_only=fit_world_only,        # 'in' or 'out' to run a single world; None = both then combine
        save_grad_norms=save_grad_norms,
        alpha=alpha,                 # significance for empirical ε estimate
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_folder = f'{args.out}/{args.data_name}_{args.model_name}_eps{args.epsilon}'
    device = args.device if torch.cuda.is_available() else 'cpu'

    # load data (define D-)
    if args.n_df == 1:
        # load single data point for type safety
        X_out, y_out, out_dim = load_data(args.data_name, 1, device=device)
    else:
        X_out, y_out, out_dim = load_data(args.data_name, args.n_df - 1, device=device)

    init_model = None
    if args.fixed_init is not None:
        init_model = Models[args.model_name](X_out.shape, out_dim=out_dim).to(device)

        if args.fixed_init == '':
            # initialize model (average-case)
            xavier_init_model(init_model)
        else:
            # load weights from path (worst-case)
            init_model.load_state_dict(torch.load(args.fixed_init))
    
    # craft target data point (x_T, y_T)
    if args.target_type == 'blank':
        # blank sample
        target_X = torch.zeros_like(X_out[[0]])
        target_y = torch.from_numpy(np.array([9])).to(device)
    elif args.target_type == 'clipbkd':
        # ClipBKD sample
        target_X, target_y = craft_clipbkd(X_out, init_model, device)
    elif os.path.exists(args.target_type):
        # pre-crafted target sample
        target_X = torch.from_numpy(np.load(args.target_type)).to(device)
        if init_model is not None:
            target_y =  choose_worstcase_label(init_model, target_X)
        else:
            target_y = torch.from_numpy(np.array([9])).to(device)
    else:
        raise Exception(f'Target {args.target_type} not found')

    # define D = D- U {(x_T, y_T)}
    X_in, y_in = torch.vstack((X_out, target_X)), torch.cat((y_out, target_y))

    # handle case where n_df = 1
    X_out, y_out = X_out[:args.n_df - 1], y_out[:args.n_df - 1]

    # load test dataset
    X_test, y_test, _ = load_data(args.data_name, None, split='test', device=device)
    
    # train M on D and D-
    # resume from checkpoint
    outputs, losses, all_grad_norms, train_set_accs, test_set_accs = resume_checkpoint(out_folder, args.save_grad_norms, args.fit_world_only, args.resume)
    worlds = [args.fit_world_only] if args.fit_world_only else ['out', 'in']
    for world in worlds:
        # set dataset according to "world"
        curr_X, curr_y = (X_out, y_out) if world == 'out' else (X_in, y_in)

        # check how many reps initially completed
        reps_completed = len(outputs[world]) 

        for rep in tqdm(range(reps_completed, args.n_reps // 2), initial=reps_completed, total=args.n_reps // 2):
            # train model
            model, grad_norms = train_model(args.model_name, curr_X, curr_y, args.epsilon, args.delta,
                args.max_grad_norm, args.n_epochs, args.lr, device=device, init_model=init_model,
                block_size=args.block_size, out_dim=out_dim)
            
            # keep track of per-sample gradient norms
            all_grad_norms[world].append(grad_norms)
            
            # get loss of model on target sample
            model.eval()
            with torch.no_grad():
                output = model(target_X)
                outputs[world].append(output[0].cpu().numpy())
                losses[world].append(-nn.CrossEntropyLoss()(output, target_y).cpu().item())
            
            # get test set accuracy from first 5 reps
            if rep < 5 and world == 'out':
                if len(X_out) > 0:
                    train_set_accs.append(test_model(model, X_out, y_out))
                test_set_accs.append(test_model(model, X_test, y_test))
            
            # free CUDA memory
            del model
            torch.cuda.empty_cache()

            # save checkpoint
            save_checkpoint(out_folder, outputs, losses, all_grad_norms, train_set_accs, test_set_accs, args.fit_world_only, args.save_grad_norms)
        outputs[world] = np.array(outputs[world])
    
    if not args.fit_world_only:
        # calculate empirical epsilon using GDP
        mia_scores = np.concatenate([losses['in'], losses['out']])
        mia_labels = np.concatenate([np.ones_like(losses['in']), np.zeros_like(losses['out'])])
        _, emp_eps_loss = compute_eps_lower_from_mia(mia_scores, mia_labels, args.alpha, args.delta, 'GDP', n_procs=1)

        np.save(f'{out_folder}/emp_eps_loss.npy', [emp_eps_loss])
        np.save(f'{out_folder}/mia_scores.npy', mia_scores)
        np.save(f'{out_folder}/mia_labels.npy', mia_labels)
    
        print(f'Theoretical eps: {args.epsilon}')
        print(f'Empirical eps: {emp_eps_loss}')

    print(f'Train set accuracy: {np.mean(train_set_accs) * 100:.3f}%')
    print(f'Test set accuracy: {np.mean(test_set_accs) * 100:.3f}%')
    return losses['in'], losses['out']


###############################################
# trained and grid search
###############################################

def convert(measure, alpha, delta):
    return measure + np.log(1/delta)/(alpha - 1)

_fixed_init = "pretrained_models/cnn_mnist_half.pt"


ALPHAS                = [2, 3, 5, 10, 20]
DV_EPOCHS             = [300, 600, 1000, 2000, 5000]
DV_HIDDEN_SETS        = [[128], [128,128], [256], [256, 256], [512, 256]]
DV_LRS                = [1e-5, 3e-5, 1e-4, 1e-3]
DV_MB_SIZES           = [128, 256, 512]
EPSILONS = [1,2,4,6,10]

OUT_CSV = "renyi_results.csv"
FIELDS = [
    "epsilon", "alpha", "epochs", "hidden", "lr", "mb",
    "mean_PQ", "max_PQ", "min_PQ",
    "mean_QP", "max_QP", "min_QP",
]
_fixed_init = "pretrained_models/cnn_mnist_half.pt"


file_exists = os.path.exists(OUT_CSV)
with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not file_exists:
        writer.writeheader()

    for epsilon in EPSILONS:
        P, Q = train_models(data_name="mnist",          # mnist | cifar10 | cifar100
            model_name="cnn",            # try 'cnn' if you crafted a CNN init
            n_reps=1000,                 # total models (split across IN/OUT worlds)
            n_df=0,                     # 0 => full dataset
            n_epochs=100,
            lr=0.01,
            max_grad_norm=1.0,
            epsilon=epsilon,                # target ε (Opacus/PRV)
            delta=1e-2,
            target_type="clipbkd",      # 'blank' | 'clipbkd' | path to a target sample
            seed=0,
            out="exp_data",
            device=_pick_device(),      # "mps" on Apple GPU, else "cuda:0" or "cpu"
            fixed_init="pretrained_models/cnn_mnist_half.pt",     # path => worst-case; '' => fixed random; None => average-case Xavier
            block_size=1000,
            resume=True,                # skip work if results are already present
            fit_world_only=None,        # 'in' or 'out' to run a single world; None = both then combine
            save_grad_norms=False,
            alpha=0.05)

        N= 1000
        for alpha in ALPHAS:
            for epochs in DV_EPOCHS:
                for hidden in DV_HIDDEN_SETS:
                    for lr in DV_LRS:
                        for mb in DV_MB_SIZES:
                            renyi_measures_PQ = []
                            renyi_measures_QP = []
                            for i in range(50):
                                tf1.compat.v1.reset_default_graph()  # avoid graph accumulation
                                renyi_measures_PQ.append(renyi_estimate(alpha, lr, epochs, hidden, mb, P, Q, N))
                                tf1.compat.v1.reset_default_graph()  # avoid graph accumulation
                                renyi_measures_QP.append(renyi_estimate(alpha, lr, epochs, hidden, mb, Q, P, N))
                            
                            renyi_measures_PQ = np.array(renyi_measures_PQ, dtype=float)
                            renyi_measures_QP = np.array(renyi_measures_QP, dtype=float)
                            mean_PQ = convert(renyi_measures_PQ.mean(), alpha, 1e-2)
                            max_PQ = convert(renyi_measures_PQ.max(), alpha, 1e-2)
                            min_PQ = convert(renyi_measures_PQ.min(), alpha, 1e-2)
                            mean_QP = convert(renyi_measures_QP.mean(), alpha, 1e-2)
                            max_QP = convert(renyi_measures_QP.max(), alpha, 1e-2)
                            min_QP = convert(renyi_measures_QP.min(), alpha, 1e-2)
                            writer.writerow({
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "epochs": epochs,
                            # lists don't serialize cleanly to CSV—store as JSON string
                            "hidden": json.dumps(hidden),
                            "lr": lr,
                            "mb": mb,
                            "mean_PQ": mean_PQ, "max_PQ": max_PQ, "min_PQ": min_PQ,
                            "mean_QP": mean_QP, "max_QP": max_QP, "min_QP": min_QP,
                            })
#def renyi_estimate(alpha, Lrate, epochs, hidden_layers, P, Q, N):
