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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='mnist', help='dataset to use (mnist, cifar10, cifar100)')
    parser.add_argument('--model_name', type=str, default='lr', choices=list(Models.keys()), help='model to audit')
    parser.add_argument('--n_reps', type=int, default=200, help='number of models')
    parser.add_argument('--n_df', type=int, default=0, help='|D| (0 => use full dataset)')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='gradient clipping norm')
    parser.add_argument('--epsilon', type=float, default=None, help='privacy parameter, epsilon')
    parser.add_argument('--delta', type=float, default=1e-5, help='privacy parameter, delta')
    parser.add_argument('--target_type', type=str, default='blank', help='sample to use as target (blank, clipbkd, or path to target sample)')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--out', type=str, default='exp_data/', help='folder to write results to')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device to use (cpu, cuda:X)')
    parser.add_argument('--fixed_init', type=str, nargs='?', default=None, const='', help='initialize all models to the same weights (if path provided, weights loaded from path (worst-case), else fix to some randomly chosen weights)')
    parser.add_argument('--block_size', type=int, default=1000, help='process samples within a batch in blocks to conserve GPU space')
    parser.add_argument('--resume', action='store_true', help='skip experiment if results are present')
    parser.add_argument('--fit_world_only', type=str, default=None, choices=['in', 'out'], help='just fit models in world and calculate losses')
    parser.add_argument('--save_grad_norms', action='store_true', help='save gradient norms for all samples in the dataset for each epoch')
    parser.add_argument('--alpha', type=float, default=0.05, help='significance level for empirical eps estimation')
    args = parser.parse_args()

    # reproducibility
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
