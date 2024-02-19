import os
import torch
import wandb
import pickle
import numpy as np
import scipy.io as sio
from dataset.dataset import Dataset
from torch.utils.data import DataLoader
from data.imca import generate_synthetic_data, ConditionalDataset
from metrics.mcc import mean_corr_coef, mean_corr_coef_per_segment
from metrics.mmse import MMSE
from model.ivae.ivae_core import iVAE
from model.ivae.ivae_wrapper import IVAE_wrapper_
from model.MISAK import MISA
from model.misa_wrapper import MISA_wrapper_
from scipy.stats import loguniform
from data.utils import to_one_hot
from sklearn.decomposition import PCA

def run_ivae_exp(args, config):

    # wandb.init(project="diva", entity="deepmisa")

    """run iVAE simulations"""
    method = args.method
    n_modality = config.n_modalities
    experiment = config.experiment
    
    # iVAE config
    if args.n_sources:
        data_dim = args.n_sources
    else:
        data_dim = config.data_dim
    
    if args.n_segments:
        n_segments = args.n_segments
    else:
        n_segments = config.n_segments
    
    if args.n_obs_per_seg:
        n_obs_per_seg = [args.n_obs_per_seg]
    else:
        n_obs_per_seg = config.n_obs_per_seg
    
    if args.ivae_batch_size:
        batch_size_ivae = args.ivae_batch_size
    else:
        batch_size_ivae = config.ivae.batch_size

    n_layers = config.n_layers
    data_seed = config.data_seed
    cuda = config.ivae.cuda
    device = config.device
    dataset = config.dataset

    # MISA config
    input_dim = [data_dim] * n_modality
    output_dim = [data_dim] * n_modality
    subspace = config.subspace
    if subspace.lower() == 'iva':
        subspace = [torch.eye(dd, device=device) for dd in output_dim]

    eta = config.eta
    beta = config.beta
    lam = config.lam
    if len(eta) > 0:
        eta = torch.tensor(eta, dtype=torch.float32, device=device)
        if len(eta) == 1:
            eta = eta*torch.ones(subspace[0].size(-2), device=device)
    else:
        # should error
        pass
    if len(beta) > 0:
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
        if len(beta) == 1:
            beta = beta*torch.ones(subspace[0].size(-2), device=device)
    else:
        # should error
        pass
    if len(lam) > 0:
        lam = torch.tensor(lam, dtype=torch.float32, device=device)
        if len(lam) == 1:
            lam = lam*torch.ones(subspace[0].size(-2), device=device)
    else:
        # should error
        pass
    
    ### TODO update code!
    if args.n_epochs:
        n_epochs = args.n_epochs
    else:
        n_epochs = config.n_epochs

    if args.ivae_lr:
        lr_ivae = args.ivae_lr
    else:
        lr_ivae = config.ivae.lr

    if args.ivae_max_iter_per_epoch:
        mi_ivae = args.ivae_max_iter_per_epoch
    else:
        mi_ivae = config.ivae.max_iter_per_epoch

    if args.misa_batch_size:
        batch_size_misa = args.misa_batch_size
    else:
        batch_size_misa = config.misa.batch_size

    seed = args.seed
    index = slice(0, n_modality)
    data_path = args.data_path

    epoch_interval = 100 # save result every n epochs
    res_corr = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epochs//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layers}
    res_recovered_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epochs//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layers}
    res_ground_truth_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epochs//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layers}
    res_metric = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epochs//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layers}
    res_model_weight = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epochs//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layers}

    for l in n_layers:
        for n in n_obs_per_seg:
            
            if experiment == "sim":
                # generate synthetic data
                x, y, s = generate_synthetic_data(data_dim, n_segments, n, l, seed=data_seed,
                    simulationMethod=dataset, one_hot_labels=True, varyMean=False)
                initial_weights = []
            
            elif experiment == "img":
                # TODO implement MGPCA
                # TODO scale covariance matrix to have equal weight
                # TODO reconstruct W 44318 x 30 x 2
                data = sio.loadmat(data_path)
                x_orig = data['x'] # 2907 x 44318 x 2
                w = data['w'] # 15 x 44318 x 2
                x = np.concatenate([np.expand_dims(x_orig[:,:,0] @ w[:,:,0].T, axis=2), np.expand_dims(x_orig[:,:,1] @ w[:,:,1].T, axis=2)], axis=2)
                initial_weights = []
                # initial_weights = [ np.eye(data_dim) for _ in range(n_modality) ]

                # x_cat = np.concatenate([x_orig[:,:,0], x_orig[:,:,1]], axis=0) # 5814 x 44318
                # pca = PCA(n_components=data_dim)
                # x_pca = pca.fit_transform(x_cat) # 5814 x 30
                # x = np.concatenate([np.expand_dims(x_pca[:x_orig.shape[0],:], axis=2), np.expand_dims(x_pca[x_orig.shape[0]:,:], axis=2)], axis=2) # 2907 x 30 x 2

                u = data['u'] 
                y = to_one_hot(u)[0] # 2907 x 14
            
            # x dimension: 4000 samples x 10 sources x 2 modalities; y dimension: 4000 samples x 20 one-hot encoding labels                    
            lr_misa = lr_ivae/n_segments
            mi_misa = mi_ivae
            
            print(f'Running {method} experiment with L={l}; n_obs_per_seg={n}; n_seg={n_segments}; n_source={data_dim}; seed={seed}; n_epochs={n_epochs}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
            
            loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}

            # TODO optimize workflow, move code block to a wrapper function
            if method.lower() == 'diva':
                model_iVAE_list = []
                ckpt_file_list = []

                # initiate iVAE model for each modality
                for m in range(n_modality):
                    ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_diva_ivae_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_seed{seed}_modality{m+1}_epoch{n_epochs}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
                    ckpt_file_list.append(ckpt_file)
                    ds = ConditionalDataset(x[:,:,m].astype(np.float32), y.astype(np.float32), device)
                    train_loader = DataLoader(ds, shuffle=True, batch_size=batch_size_ivae, **loader_params)
                    data_dim, latent_dim, aux_dim = ds.get_dims() # data_dim = 30, latent_dim = 30, aux_dim = 14
                    
                    model_iVAE = iVAE(latent_dim, 
                                    data_dim, 
                                    aux_dim, 
                                    activation='lrelu', 
                                    device=device, 
                                    n_layers=l, 
                                    hidden_dim=data_dim * 2,
                                    method=method.lower(),
                                    seed=seed)
                    
                    res_iVAE, model_iVAE, params_iVAE = IVAE_wrapper_(X=x[:,:,m], U=y, batch_size=batch_size_ivae, 
                                                                      n_layers=l, hidden_dim=data_dim * 2, 
                                                                      cuda=cuda, max_iter=mi_ivae, lr=lr_ivae,
                                                                      ckpt_file=ckpt_file, seed=seed, model=model_iVAE) #model=model_MISA.input_model[m]
                    
                    res_ivae = res_iVAE.detach().numpy()
                    fname = os.path.join(args.run, f'src_ivae_m{m+1}_{experiment}_{method.lower()}_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                    pickle.dump(res_ivae, open(fname, "wb"))

                    model_iVAE_list.append(model_iVAE)
                
                for m in range(n_modality):
                    model_iVAE_list[m].set_aux(False)
                    # print(f"model_iVAE_list[{m}].use_aux = {model_iVAE_list[m].use_aux}")

                model_MISA = MISA(weights=initial_weights, # TODO MGPCA weights "mgpca"
                                    index=index, 
                                    subspace=subspace, 
                                    eta=eta, 
                                    beta=beta, 
                                    lam=lam, 
                                    input_dim=input_dim, 
                                    output_dim=output_dim, 
                                    seed=seed, 
                                    device=device,
                                    model=model_iVAE_list)

                ckpt_file_misa = os.path.join(args.run, 'checkpoints', f'{experiment}_diva_misa_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_seed{seed}_modality{m+1}_epoch{n_epochs}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')

                # update iVAE and MISA model weights
                # run iVAE per modality
                np.random.seed(seed)
                rand_seq = np.random.randint(0, 1000, size=n_epochs*mi_misa)

                for e in range(n_epochs):
                    print('Epoch: {}'.format(e))
                    # loop MISA through segments
                    # remove the mean of segment because MISA loss assumes zero mean
                    # randomize segment order
                    for it in range(mi_misa):
                        np.random.seed(rand_seq[e*mi_misa+it])
                        segment_shuffled = np.arange(n_segments)
                        np.random.shuffle(segment_shuffled)

                        for seg in segment_shuffled:
                            if experiment == "sim":
                                y_seg = y[seg*n:(seg+1)*n]
                                x_seg = x[seg*n:(seg+1)*n,:,:]
                            elif experiment == "img":
                                ind = np.where(y[:,seg]==1)[0]
                                y_seg = y[ind,:]
                                x_seg = x[ind,:,:]

                            x_seg_dm = x_seg - np.mean(x_seg, axis=0) # remove mean of segment
                            
                            ds = ConditionalDataset(x_seg_dm.astype(np.float32), y_seg.astype(np.float32), device)
                            train_loader = DataLoader(ds, shuffle=True, batch_size=batch_size_misa, **loader_params)
                            test_loader = DataLoader(ds, shuffle=False, batch_size=len(ds), **loader_params)

                            model_MISA, final_MISI = MISA_wrapper_(data_loader=train_loader,
                                                test_data_loader=test_loader,
                                                epochs=1,
                                                lr=lr_misa,
                                                device=device,
                                                ckpt_file=ckpt_file_misa,
                                                model_MISA=model_MISA)
                            
                    if e % epoch_interval == 0:
                        for mm in range(n_modality):
                            for ll in range(l):
                                res_model_weight[l][n][e].append(model_MISA.input_model[mm].g.fc[ll].weight)

                    # train iVAE for 1000x more iterations
                    # if e == n_epochs - 1:
                    #     mi_ivae = mi_ivae * 1000

                    for m in range(n_modality):
                        model_MISA.input_model[m].set_aux(True)
                        # print(f"model_MISA.input_model[{m}].use_aux = {model_MISA.input_model[m].use_aux}")
                        
                        res_iVAE, model_MISA.input_model[m], params_iVAE = IVAE_wrapper_(X=x[:,:,m], U=y, batch_size=batch_size_ivae, 
                                                                                         n_layers=n_layers, hidden_dim=data_dim * 2, 
                                                                                         cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, 
                                                                                         ckpt_file=ckpt_file_list[m], seed=seed, 
                                                                                         test=False, model=model_MISA.input_model[m])
                        
                        model_MISA.input_model[m].set_aux(False)
                        # print(f"model_MISA.input_model[{m}].use_aux = {model_MISA.input_model[m].use_aux}")

                        # store results every epoch_interval epochs
                        if e % epoch_interval == 0:
                            res_ivae = res_iVAE.detach().numpy()
                            
                            if experiment == 'sim':
                                res_corr[l][n][e].append(mean_corr_coef(res_ivae, s[:,:,m]))
                                res_corr[l][n][e].append(mean_corr_coef_per_segment(res_ivae, s[:,:,m], y))
                                print(res_corr[l][n][e][0])
                                res_recovered_source[l][n][e].append(res_ivae)
                                
                                if m == n_modality - 1: # last epoch, last modality
                                    res_ground_truth_source[l][n][e].append(s)
                                    res_ivae_stack = np.dstack(res_recovered_source[l][n][e])
                                    metric = MMSE(res_ivae_stack, s, y)
                                    res_metric[l][n][e].append(metric)

                                    res = {
                                        'mcc': res_corr[l][n][e],
                                        'recovered_source': res_recovered_source[l][n][e],
                                        'ground_truth_source': res_ground_truth_source[l][n][e],
                                        'metric': res_metric[l][n][e],
                                        'model_weight': res_model_weight[l][n][e]
                                    }

                                    fname = os.path.join(args.run, f'res_{experiment}_{method.lower()}_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                    pickle.dump(res, open(fname, "wb"))

                            elif experiment == 'img':
                                res_recovered_source[l][n][e].append(res_ivae)

                                if m == n_modality - 1:
                                    res = {
                                        'recovered_source': res_recovered_source[l][n][e]
                                    }

                                    fname = os.path.join(args.run, f'res_{experiment}_{method.lower()}_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                    pickle.dump(res, open(fname, "wb"))

            elif method.lower() == 'ivae':
                # intiate iVAE model for each modality
                model_iVAE_list = []
                
                for m in range(n_modality):
                    ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_ivae_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_seed{seed}_modality{m+1}_epoch{n_epochs}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
                    ds = ConditionalDataset(x[:,:,m].astype(np.float32), y.astype(np.float32), device)
                    train_loader = DataLoader(ds, shuffle=True, batch_size=batch_size_ivae, **loader_params)
                    data_dim, latent_dim, aux_dim = ds.get_dims() # data_dim: 10, latent_dim: 10, aux_dim: 20
                    
                    model_iVAE = iVAE(latent_dim, 
                                    data_dim, 
                                    aux_dim, 
                                    activation='lrelu', 
                                    device=device, 
                                    n_layers=l, 
                                    hidden_dim=data_dim * 2,
                                    method=method.lower(),
                                    seed=seed)
                    
                    for e in range(n_epochs):
                        print('Epoch: {}'.format(e))
                        res_iVAE, model_iVAE, params_iVAE = IVAE_wrapper_(X=x[:,:,m], U=y, batch_size=batch_size_ivae, 
                                                                          n_layers=l, hidden_dim=data_dim * 2, 
                                                                          cuda=cuda, max_iter=mi_ivae, lr=lr_ivae,
                                                                          ckpt_file=ckpt_file, seed=seed, model=model_iVAE)
                        
                        if e % epoch_interval == 0:
                            res_ivae = res_iVAE.detach().numpy()
                            
                            if experiment == 'sim':
                                res_corr[l][n][e].append(mean_corr_coef(res_ivae, s[:,:,m]))
                                res_corr[l][n][e].append(mean_corr_coef_per_segment(res_ivae, s[:,:,m], y))
                                print(res_corr[l][n][e][0])
                                res_recovered_source[l][n][e].append(res_ivae)
                            
                                if m == n_modality - 1: # last epoch, last modality
                                    res_ground_truth_source[l][n][e].append(s)
                                    res_ivae_stack = np.dstack(res_recovered_source[l][n][e])
                                    metric = MMSE(res_ivae_stack, s, y)
                                    res_metric[l][n][e].append(metric)

                                    res = {
                                        'mcc': res_corr[l][n][e],
                                        'recovered_source': res_recovered_source[l][n][e],
                                        'ground_truth_source': res_ground_truth_source[l][n][e],
                                        'metric': res_metric[l][n][e]
                                    }

                                    fname = os.path.join(args.run, f'res_{experiment}_{method.lower()}_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                    pickle.dump(res, open(fname, "wb"))
                            
                            elif experiment == 'img':
                                res_recovered_source[l][n][e].append(res_ivae)
                                
                                if m == n_modality - 1:
                                    res = {
                                        'recovered_source': res_recovered_source[l][n][e]
                                    }

                                    fname = os.path.join(args.run, f'res_{experiment}_{method.lower()}_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                    pickle.dump(res, open(fname, "wb"))
                                
                    model_iVAE_list.append(model_iVAE)
                    
            elif method.lower() == 'misa':
                ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_misa_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_seed{seed}_epoch{n_epochs}_maxiter{mi_misa}_lrmisa{round(lr_misa, 5)}.pt')
                
                # initialize MISA model weights using iVAE sources as A = (z^T z)^{-1} z^T X
                for m in range(n_modality):
                    fname = os.path.join(args.run, f'src_ivae_m{m+1}_{experiment}_diva_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                    if os.path.exists(fname):
                        print(f'Loading iVAE source {fname}')
                        res_ivae = pickle.load(open(fname, 'rb'))
                        weight_init = np.linalg.inv(res_ivae.T @ res_ivae) @ res_ivae.T @ x[:,:,m]
                        initial_weights.append(weight_init)

                model_MISA = MISA(weights=initial_weights,
                    index=index, 
                    subspace=subspace, 
                    eta=eta, 
                    beta=beta, 
                    lam=lam, 
                    input_dim=input_dim, 
                    output_dim=output_dim, 
                    seed=seed, 
                    device=device)

                # update iVAE and MISA model weights
                # run iVAE per modality
                np.random.seed(seed)
                segment_shuffled = np.arange(n_segments)
                np.random.shuffle(segment_shuffled)

                if experiment == "sim":
                    res_MISA = np.zeros_like(s)
                else:
                    res_MISA = np.zeros((x.shape[0], data_dim, n_modality))

                for e in range(n_epochs):
                    print('Epoch: {}'.format(e))
                    # loop MISA through segments
                    # remove the mean of segment because MISA loss assumes zero mean
                    # randomize segment order
                    for seg in segment_shuffled:
                        if experiment == "sim":
                            x_seg = x[seg*n:(seg+1)*n,:,:]
                        elif experiment == "img":
                            ind = np.where(y[:,seg]==1)[0]
                            x_seg = x[ind,:,:]
                        x_seg_dm = x_seg - np.mean(x_seg, axis=0) # remove mean of segment
                        # a list of datasets, each dataset dimension is sample x source
                        ds = Dataset(data_in=x_seg_dm, device=device)
                        train_loader = DataLoader(dataset=ds, batch_size=batch_size_misa, shuffle=True)
                        test_loader = DataLoader(dataset=ds, batch_size=len(ds), shuffle=False)

                        model_MISA, final_MISI = MISA_wrapper_(data_loader=train_loader,
                                            test_data_loader=test_loader,
                                            epochs=mi_misa,
                                            lr=lr_misa,
                                            device=device,
                                            ckpt_file=ckpt_file,
                                            model_MISA=model_MISA)

                        if e % epoch_interval == 0:
                            for m in range(n_modality):
                                if experiment == "sim":
                                    res_MISA[seg*n:(seg+1)*n,:,m] = model_MISA.output[m].detach().numpy()
                                elif experiment == "img":
                                    res_MISA[ind,:,m] = model_MISA.output[m].detach().numpy()
                    
                    if e % epoch_interval == 0:
                        for m in range(n_modality):
                            if experiment == 'sim':
                                res_corr[l][n][e].append(mean_corr_coef(res_MISA[:,:,m], s[:,:,m]))
                                res_corr[l][n][e].append(mean_corr_coef_per_segment(res_MISA[:,:,m], s[:,:,m], y))
                                print(res_corr[l][n][e][0])
                                res_recovered_source[l][n][e].append(res_MISA[:,:,m])
                                
                                if m == n_modality - 1: # last epoch, last modality
                                    res_ground_truth_source[l][n][e].append(s)
                                    metric = MMSE(res_MISA, s, y)
                                    res_metric[l][n][e].append(metric)

                                    res = {
                                        'mcc': res_corr[l][n][e],
                                        'recovered_source': res_recovered_source[l][n][e],
                                        'ground_truth_source': res_ground_truth_source[l][n][e],
                                        'metric': res_metric[l][n][e]
                                    }

                                    fname = os.path.join(args.run, f'res_{experiment}_{method.lower()}_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                    pickle.dump(res, open(fname, "wb"))

                            elif experiment == 'img':
                                res_recovered_source[l][n][e].append(res_MISA[:,:,m])
                                
                                if m == n_modality - 1:
                                    res = {
                                        'recovered_source': res_recovered_source[l][n][e]
                                    }

                                    fname = os.path.join(args.run, f'res_{experiment}_{method.lower()}_layer{l}_source{data_dim}_obs{n}_seg{n_segments}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                    pickle.dump(res, open(fname, "wb"))
    
    # prepare output
    Results = {
        'mcc': res_corr,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'metric': res_metric
    }

    return Results
