import wandb
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from data.imca import ConditionalDataset
from .ivae_core import iVAE

def IVAE_wrapper(X, U, batch_size=256, max_iter=7e4, n_layer=3, hidden_dim=20, lr=1e-3, 
                  seed=0, cuda=True, ckpt_file='ivae.pt', test=False, model=None, Xv=None, Uv=None):
    " args are the arguments from the main.py file"
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load data
    dset = ConditionalDataset(X.astype(np.float32), U.astype(np.float32), device)
    loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    data_loader = DataLoader(dset, shuffle=True, batch_size=batch_size, **loader_params)
    valid = False
    if (Xv is not None) and (Uv is not None):
        valid = True
        valid_dset = ConditionalDataset(Xv.astype(np.float32), Uv.astype(np.float32), device)
        valid_data_loader = DataLoader(valid_dset, shuffle=False, batch_size=batch_size, **loader_params)

    # training loop
    if not test:
        # define optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)

        model.train()
        for it in range(max_iter):
            elbo_train = 0
            for _, (x, u) in enumerate(data_loader):
                optimizer.zero_grad()
                x, u = x.to(device), u.to(device)
                elbo, z_est = model.elbo(x, u)
                elbo.mul(-1).backward()
                optimizer.step()
                elbo_train += -elbo.item()
            elbo_train /= len(data_loader)
            scheduler.step(elbo_train)

            if valid:
                elbo_valid = 0
                for _, (x, u) in enumerate(valid_data_loader):
                    x, u = x.to(device), u.to(device)
                    elbo, z_est = model.elbo(x, u)
                    elbo_valid += -elbo.item()
                elbo_valid /= len(valid_data_loader)

                print(f'iVAE training loss: {elbo_train:.3f}; validation loss: {elbo_valid:.3f}')
                wandb.log({'iVAE training loss': elbo_train, 'iVAE validation loss': elbo_valid})
        # save model checkpoint after training
        torch.save(model.state_dict(), ckpt_file)
    else:
        model_params = torch.load(ckpt_file, map_location=device)
        # TODO verify
        with torch.no_grad():
            for l in range(n_layer):
                model.logl.fc[l].weight.copy_(model_params[f'logl.fc.{l}.weight'])
                model.logl.fc[l].bias.copy_(model_params[f'logl.fc.{l}.bias'])
                model.f.fc[l].weight.copy_(model_params[f'f.fc.{l}.weight'])
                model.f.fc[l].bias.copy_(model_params[f'f.fc.{l}.bias'])
                model.g.fc[l].weight.copy_(model_params[f'g.fc.{l}.weight'])
                model.g.fc[l].bias.copy_(model_params[f'g.fc.{l}.bias'])
                model.logv.fc[l].weight.copy_(model_params[f'logv.fc.{l}.weight'])
                model.logv.fc[l].bias.copy_(model_params[f'logv.fc.{l}.bias'])
        elbo_test = 0
        for _, (x, u) in enumerate(data_loader):
            x, u = x.to(device), u.to(device)
            elbo, z_est = model.elbo(x, u)
            elbo_test += -elbo.item()
        elbo_test /= len(data_loader)
        print(f'iVAE test loss: {elbo_test:.3f}')

    Xt, Ut = dset.x, dset.y
    decoder_params, encoder_params, zt, prior_params = model(Xt, Ut)
    params = {'decoder': decoder_params, 'encoder': encoder_params, 'prior': prior_params}
    
    if valid:
        Xv, Uv = valid_dset.x, valid_dset.y
        _, _, zv, _ = model(Xv, Uv)
        z = [zt, zv]
    else:
        z = zt
    
    return z, model, params