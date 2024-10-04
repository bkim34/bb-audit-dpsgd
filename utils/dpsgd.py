"""
Utility functions to execute DP-SGD
"""
import torch
import numpy as np
from torch.func import functional_call, vmap, grad

def get_per_sample_grads(model, X, y, criterion):
    """Compute per-sample gradients"""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch,))
        loss = criterion(predictions, targets)
        return loss
    
    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    ps_grads = ft_compute_sample_grad(params, buffers, X, y)

    return ps_grads

def get_per_sample_grad_norms(per_sample_grads):
    """Compute L2 norms of per-sample gradients"""
    return torch.vstack([
        curr_grad.flatten(start_dim=1).norm(2, dim=1)
        for curr_grad in per_sample_grads.values()
    ]).norm(2, dim=0)

def clip_per_sample_grads(per_sample_grads, max_grad_norm):
    """Clip per-sample gradients to clipping norm"""
    ps_grad_norms = get_per_sample_grad_norms(per_sample_grads)

    ps_grad_scales = 1 / torch.maximum(
        torch.ones_like(ps_grad_norms),
        ps_grad_norms / max_grad_norm
    )

    ps_grads_clipped = {
        # broadcast
        name: curr_grad * ps_grad_scales[(...,) + (None,) * (curr_grad.dim() - 1)]
        for name, curr_grad in per_sample_grads.items() 
    }

    ps_grad_norms_clipped = get_per_sample_grad_norms(ps_grads_clipped)

    return ps_grads_clipped, { 'before': ps_grad_norms.cpu().numpy(), 'after': ps_grad_norms_clipped.cpu().numpy() }

def clip_and_accum_grads_block(model, X, y, optimizer, criterion, max_grad_norm):
    """Clip and accumulate gradients of a single block of samples"""
    optimizer.zero_grad()

    if len(X) == 0:
        # empty dataset
        ps_grads = { name: torch.zeros_like(param).unsqueeze(dim=0) for name, param in model.named_parameters() }
    else:
        # calculate per-sample gradients
        ps_grads = get_per_sample_grads(model, X, y, criterion)

    ps_grad_norms_data = { 'before': np.array([]), 'after': np.array([]) }
    if max_grad_norm is not None:
        # clip per-sample gradients
        ps_grads_clipped, ps_grad_norms_data = clip_per_sample_grads(ps_grads, max_grad_norm)
    else:
        ps_grads_clipped = ps_grads
    
    # accumulate per-sample gradients
    with torch.no_grad():
        accum_grads = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}
    
    return accum_grads, ps_grad_norms_data

def clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm, block_size=1024):
    """Clip and accumulate gradients in blocks of samples to conserve gpu space"""
    # split samples into blocks by index
    idx_blocks = torch.split(torch.from_numpy(np.arange(len(X))), block_size)

    accum_grad = None
    ps_grad_norms_data = { 'before': [], 'after': [] }
    for idx_block in idx_blocks:
        # get a single block of samples
        curr_X, curr_y = X[idx_block], y[idx_block]

        # accum grads for this single block
        accum_grad_block, curr_ps_grad_norms_data = clip_and_accum_grads_block(model, curr_X, curr_y, optimizer, criterion, max_grad_norm)
        ps_grad_norms_data['before'].append(curr_ps_grad_norms_data['before'])
        ps_grad_norms_data['after'].append(curr_ps_grad_norms_data['after'])

        # accum grads for all blocks
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name, curr_grad in accum_grad_block.items():
                    accum_grad[name] = accum_grad[name] + curr_grad
    
    ps_grad_norms_data['before'] = np.concatenate(ps_grad_norms_data['before'])
    ps_grad_norms_data['after'] = np.concatenate(ps_grad_norms_data['after'])
    
    return accum_grad, ps_grad_norms_data