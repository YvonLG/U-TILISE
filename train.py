
from pathlib import Path
import pickle
import os

import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from tqdm import tqdm

from data.dataset import TimeSeriesDS
from model.utilise import UTILISE
from model.utils import initialize_weights


# TODO: setup iid and ood validation dataset
# TODO: log prediction and attn during training
# TODO: use wandb
# TODO: setup metrics
# TODO: use SEN12 dataset
# TODO: use EARTHNET2021 dataset

model_opt = dict(
    input_nc=4+2,
    output_nc=4,
    nch=64,
    n_depth=3,
    activ='relu',
    norm='none',
    padding='zeros',
    n_groups=4,
    upsample='bilinear',
    n_heads=4,
    d_key=4,
    ffn_nc=[256],
    attn_dropout=0.1,
    dropout=0.1,
    pos_encoding=True,
    learn_encoding=True
)

dataset_opt = dict(
    desired_len=10,
    min_len=5,
    occl_range=(6, 8),
    cloud_treshold=0.5,
    cloudfree_treshold=0.01,
    fullcover_treshold=0.99,
    use_quadrants=True,
    use_padding=True,
    use_transforms=True,
    continuous_sample=True,
    s2_nch=4,
    s2_used_ch=None,
    s1_nch=2,
    s1_used_ch=None,
    seed=None
)

opt = dict(
    init_type='kaiming',
    learning_rate=2e-4,
    beta1=0.9,
    weight_decay=0.01,
    grad_clip=0,
    log_interval=25,
)
opt.update({
    'model_opt': model_opt,
    'dataset_opt': dataset_opt,
})


EPOCHS = 3
PROJECT = 'first-project'
run_id = '726vna2r'

if run_id is None:
    run = wandb.init(project=PROJECT, config=opt)
    
    with open('../data/train.pickle', 'rb') as src:
        path_dicts_train = pickle.load(src)
    
    with open('../data/valid.pickle', 'rb') as src:
        path_dicts_valid = pickle.load(src)

    # TODO: valid dataset ood

    with open(os.path.join(wandb.run.dir, 'train.pickle'), 'wb') as src:
        pickle.dump(path_dicts_train, src)

    with open(os.path.join(wandb.run.dir, 'valid.pickle'), 'wb') as src:
        pickle.dump(path_dicts_valid, src)

    wandb.save('train.pickle')
    wandb.save('valid.pickle')

    dataset_train = TimeSeriesDS(path_dicts=path_dicts_train, **dataset_opt)
    dataset_valid = TimeSeriesDS(path_dicts=path_dicts_valid, **dataset_opt)

    model = UTILISE(**model_opt)
    not_initialized = True

else:
    run = wandb.init(project=PROJECT, id=run_id, resume='allow')
    
    wandb.restore('train.pickle')
    wandb.restore('valid.pickle')
    not_initialized = wandb.restore('model.pt') is None

    with open(os.path.join(wandb.run.dir, 'train.pickle'), 'rb') as src:
        path_dicts_train = pickle.load(src)

    with open(os.path.join(wandb.run.dir, 'valid.pickle'), 'rb') as src:
        path_dicts_valid = pickle.load(src)
    
    dataset_train = TimeSeriesDS(path_dicts=path_dicts_train, **wandb.config['dataset_opt'])

    model = UTILISE(**wandb.config['model_opt'])
    if not not_initialized:
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'model.pt')))


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
if not_initialized:
    initialize_weights(model, wandb.config['init_type'])

# GPU can't handle bs > 1 anyway
dataloader_train = DataLoader(dataset_train, 1)

optim = AdamW(model.parameters(),
              lr=wandb.config['learning_rate'],
              betas=(wandb.config['beta1'], 0.999),
              weight_decay=wandb.config['weight_decay'])

criterion = nn.L1Loss().to(device)

wandb.watch(model, log='all', log_freq=40)
step = 0
for epoch in range(1, EPOCHS+1):
    
    model.train()
    train_losses = []

    for data in tqdm(dataloader_train):
        
        if len(data) == 5:
            s2, s1, masks, doy, pad_mask = data
            pad_mask = pad_mask.to(device)
        else:
            s2, s1, masks, doy = data
            pad_mask = None

        s2 = s2.to(device)
        s1 = s1.to(device)
        masks = masks.to(device)
        doy = doy.to(device)

        # occlude s2
        masks_broadcast = masks.unsqueeze(2).expand_as(s2)
        s2_occl = torch.where(masks_broadcast, 1., s2)

        s2_and_s1 = torch.cat([s2_occl, s1], dim=2)

        s2_out = model(s2_and_s1, doy, pad_mask)
        loss = criterion(s2_out[~pad_mask], s2[~pad_mask])

        optim.zero_grad()
        loss.backward()

        grad_clip = wandb.config['grad_clip']
        if grad_clip:
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), grad_clip)

        optim.step()

        train_losses.append(loss.detach().cpu().item())
        step += 1

        metrics= {}

        if step % wandb.config['log_interval'] == 0:

            train_loss = sum(train_losses) / len(train_losses)
            train_losses = []

            metrics['train'] = {'loss': train_loss}

            wandb.log(metrics)
    
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    wandb.save('model.pt')

