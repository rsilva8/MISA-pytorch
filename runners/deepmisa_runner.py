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
from model.ivae.ivae_wrapper import IVAE_wrapper
from model.MISAK import MISA
# from model.MISAKinit import MISA
from model.misa_wrapper import MISA_wrapper_
from model.icebeem_wrapper import ICEBEEM_wrapper
from data.utils import to_one_hot

def split_sim_data(x, y, s, n_segment, n_obs_per_seg):
    ind_list_train, ind_list_valid, ind_list_test = [], [], []
    for i in range(n_segment):
        ind_list_train += np.arange(i*n_obs_per_seg*3, i*n_obs_per_seg*3+n_obs_per_seg).tolist()
        ind_list_valid += np.arange(i*n_obs_per_seg*3+n_obs_per_seg, i*n_obs_per_seg*3+2*n_obs_per_seg).tolist()
        ind_list_test += np.arange(i*n_obs_per_seg*3+2*n_obs_per_seg, (i+1)*n_obs_per_seg*3).tolist()
    x_train = x[ind_list_train,:,:]
    y_train = y[ind_list_train,:]
    s_train = s[ind_list_train,:,:]
    x_valid = x[ind_list_valid,:,:]
    y_valid = y[ind_list_valid,:]
    s_valid = s[ind_list_valid,:,:]
    x_test = x[ind_list_test,:,:]
    y_test = y[ind_list_test,:]
    s_test = s[ind_list_test,:,:]
    return x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test


def split_img_data(data):
    # with MGPCA
    # x_orig = data['x'] # 2907 x 44318 x 2
    # w = data['w'] # 15 x 44318 x 2
    # x = np.concatenate([np.expand_dims(x_orig[:,:,0] @ w[:,:,0].T, axis=2), np.expand_dims(x_orig[:,:,1] @ w[:,:,1].T, axis=2)], axis=2)
    
    # without MGPCA
    x_train = data['x_train']
    x_valid = data['x_valid']
    x_test = data['x_test']
    # x_cat = np.concatenate([x_orig[:,:,0], x_orig[:,:,1]], axis=0)
    # pca = PCA(n_components=data_dim)
    # x_pca = pca.fit_transform(x_cat)
    # x = np.concatenate([np.expand_dims(x_pca[:x_orig.shape[0],:], axis=2), np.expand_dims(x_pca[x_orig.shape[0]:,:], axis=2)], axis=2) # 2907 x 30 x 2
    
    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']

    u_train = to_one_hot(data['u_train'])[0]
    u_valid = to_one_hot(data['u_valid'])[0]
    u_test = to_one_hot(data['u_test'])[0]

    return x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test


def run_diva(args, config, method="diva"):
    wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    index = slice(0, n_modality)
    experiment = config.experiment
    n_layer = config.n_layer
    dataset = config.dataset
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = [args.n_obs_per_seg] if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch

    # MISA config
    input_dim = [latent_dim] * n_modality
    output_dim = [latent_dim] * n_modality
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
    
    if len(beta) > 0:
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
        if len(beta) == 1:
            beta = beta*torch.ones(subspace[0].size(-2), device=device)
    
    if len(lam) > 0:
        lam = torch.tensor(lam, dtype=torch.float32, device=device)
        if len(lam) == 1:
            lam = lam*torch.ones(subspace[0].size(-2), device=device)
    
    batch_size_misa = args.misa_batch_size if args.misa_batch_size else config.misa.batch_size

    epoch_interval = 100 # save result every n epochs
    res_corr = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_recovered_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_ground_truth_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_metric = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_model_weight = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}

    for l in n_layer:
        for n in n_obs_per_seg:
            
            if experiment == "sim":
                # generate synthetic data
                x, y, s = generate_synthetic_data(latent_dim, n_segment, n*3, l, seed=data_seed,
                    simulationMethod=dataset, one_hot_labels=True, varyMean=False)
                
                x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n)

                initial_weights = []
            
            elif experiment == "img":
                data = sio.loadmat(data_path)
                
                x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test = split_img_data(data)

                initial_weights = []
                # initial_weights = [ np.eye(latent_dim) for _ in range(n_modality) ]
            
            lr_misa = lr_ivae/n_segment
            mi_misa = mi_ivae
            
            print(f'Running {method} experiment with L={l}; n_obs_per_seg={n}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
            
            loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}

            model_ivae_list = []
            ckpt_file_list = []
            data_dim = x_train.shape[1]
            aux_dim = y_train.shape[1]

            # initiate iVAE model for each modality
            for m in range(n_modality):
                ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_ivae_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_seed{seed}_modality{m+1}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
                ckpt_file_list.append(ckpt_file)
                
                model_ivae = iVAE(latent_dim, 
                                data_dim, 
                                aux_dim, 
                                activation='lrelu', 
                                device=device, 
                                n_layer=l, 
                                hidden_dim=latent_dim * 2,
                                method=method,
                                seed=seed)
                print('\n')
                [res_ivae_train, res_ivae_valid], model_ivae, params_ivae = IVAE_wrapper(X=x_train[:,:,m], U=y_train, batch_size=batch_size_ivae, 
                                                                    n_layer=l, hidden_dim=latent_dim * 2,
                                                                    cuda=cuda, max_iter=mi_ivae, lr=lr_ivae,
                                                                    ckpt_file=ckpt_file, seed=seed, model=model_ivae, 
                                                                    Xv=x_valid[:,:,m], Uv=y_valid) #model=model_misa.input_model[m]
                
                res_ivae_train = res_ivae_train.detach().numpy()
                fname = os.path.join(args.run, f'src_ivae_m{m+1}_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                pickle.dump(res_ivae_train, open(fname, "wb"))

                model_ivae_list.append(model_ivae)
            
            for m in range(n_modality):
                model_ivae_list[m].set_aux(False)
                # print(f"model_ivae_list[{m}].use_aux = {model_ivae_list[m].use_aux}")

            model_misa = MISA(weights=initial_weights,
                                index=index, 
                                subspace=subspace, 
                                eta=eta, 
                                beta=beta, 
                                lam=lam, 
                                input_dim=input_dim, 
                                output_dim=output_dim, 
                                seed=seed, 
                                device=device,
                                model=model_ivae_list,
                                latent_dim=latent_dim)

            ckpt_file_misa = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_misa_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_seed{seed}_modality{m+1}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')

            # update iVAE and MISA model weights
            # run iVAE per modality
            np.random.seed(seed)
            rand_seq = np.random.randint(0, 1000, size=n_epoch*mi_misa)

            for e in range(n_epoch):
                print(f'\nEpoch: {e}')
                print('\n')
                # loop MISA through segments
                # remove the mean of segment because MISA loss assumes zero mean
                # randomize segment order
                for it in range(mi_misa):
                    np.random.seed(rand_seq[e*mi_misa+it])
                    segment_shuffled = np.arange(n_segment)
                    np.random.shuffle(segment_shuffled)

                    for seg in segment_shuffled:
                        if experiment == "sim":
                            y_seg_train = y_train[seg*n:(seg+1)*n]
                            x_seg_train = x_train[seg*n:(seg+1)*n,:,:]
                            y_seg_valid = y_valid[seg*n:(seg+1)*n]
                            x_seg_valid = x_valid[seg*n:(seg+1)*n,:,:]
                        elif experiment == "img":
                            ind_train = np.where(u_train[:,seg]==1)[0]
                            y_seg_train = y_train[ind_train,:]
                            x_seg_train = x_train[ind_train,:,:]

                            ind_valid = np.where(u_valid[:,seg]==1)[0]
                            y_seg_valid = y_valid[ind_valid,:]
                            x_seg_valid = x_valid[ind_valid,:,:]

                        x_seg_dm_train = x_seg_train - np.mean(x_seg_train, axis=0) # remove mean of segment
                        x_seg_dm_valid = x_seg_valid - np.mean(x_seg_valid, axis=0) # remove mean of segment
                        
                        ds_train = ConditionalDataset(x_seg_dm_train.astype(np.float32), y_seg_train.astype(np.float32), device)
                        ds_valid = ConditionalDataset(x_seg_dm_valid.astype(np.float32), y_seg_valid.astype(np.float32), device)
                        data_loader_train = DataLoader(ds_train, shuffle=True, batch_size=len(ds_train), **loader_params)
                        data_loader_valid = DataLoader(ds_valid, shuffle=False, batch_size=len(ds_valid), **loader_params)

                        model_misa, output_MISA, _ = MISA_wrapper_(data_loader=data_loader_train,
                                            test_data_loader=data_loader_valid,
                                            epochs=1,
                                            lr=lr_misa,
                                            device=device,
                                            ckpt_file=ckpt_file_misa,
                                            model_MISA=model_misa)
                        
                if e % epoch_interval == 0:
                    for mm in range(n_modality):
                        for ll in range(l):
                            res_model_weight[l][n][e].append(model_misa.input_model[mm].g.fc[ll].weight)

                for m in range(n_modality):
                    model_misa.input_model[m].set_aux(True)
                    # print(f"model_misa.input_model[{m}].use_aux = {model_misa.input_model[m].use_aux}")
                    print('\n')
                    [res_ivae_train, res_ivae_valid], model_misa.input_model[m], params_ivae = IVAE_wrapper(X=x_train[:,:,m], U=y_train, batch_size=batch_size_ivae, 
                                                                                    n_layer=l, hidden_dim=latent_dim * 2, 
                                                                                    cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, 
                                                                                    ckpt_file=ckpt_file_list[m], seed=seed, 
                                                                                    test=False, model=model_misa.input_model[m],
                                                                                    Xv=x_valid[:,:,m], Uv=y_valid)
                    
                    model_misa.input_model[m].set_aux(False)
                    # print(f"model_misa.input_model[{m}].use_aux = {model_misa.input_model[m].use_aux}")

                    # store test results every epoch_interval epochs
                    if e % epoch_interval == 0:
                        print('\n')
                        res_ivae, _, _ = IVAE_wrapper(X=x_test[:,:,m], U=y_test, batch_size=batch_size_ivae, 
                                                    n_layer=l, hidden_dim=latent_dim * 2, 
                                                    cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, 
                                                    ckpt_file=ckpt_file_list[m], seed=seed, 
                                                    test=True, model=model_misa.input_model[m])
                        
                        res_ivae = res_ivae.detach().numpy()
                        res_ivae_train = res_ivae_train.detach().numpy()
                        res_ivae_valid = res_ivae_valid.detach().numpy()
                        res_recovered_source[l][n][e].append({'test':res_ivae, 'train':res_ivae_train, 'valid':res_ivae_valid})
                        
                        if experiment == 'sim':
                            res_corr[l][n][e].append(mean_corr_coef(res_ivae, s_test[:,:,m]))
                            res_corr[l][n][e].append(mean_corr_coef_per_segment(res_ivae, s_test[:,:,m], y_test))
                            print(res_corr[l][n][e][0])
                            
                            if m == n_modality - 1: # last epoch, last modality
                                res_ground_truth_source[l][n][e].append({'test':s_test, 'train':s_train, 'valid':s_valid})
                                res_ivae_stack = np.dstack([r['test'] for r in res_recovered_source[l][n][e]])
                                metric = MMSE(res_ivae_stack, s_test, y_test)
                                res_metric[l][n][e].append(metric)

                                res = {
                                    'mcc': res_corr[l][n][e],
                                    'recovered_source': res_recovered_source[l][n][e],
                                    'ground_truth_source': res_ground_truth_source[l][n][e],
                                    'metric': res_metric[l][n][e],
                                    'model_weight': res_model_weight[l][n][e]
                                }

                                fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{data_dim}_obs{n}_seg{n_segment}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                pickle.dump(res, open(fname, "wb"))

                        elif experiment == 'img':
                            if m == n_modality - 1:
                                res = {
                                    'recovered_source': res_recovered_source[l][n][e]
                                }

                                fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{data_dim}_obs{n}_seg{n_segment}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                pickle.dump(res, open(fname, "wb"))
    
    # prepare output
    Results = {
        'mcc': res_corr,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'metric': res_metric
    }

    return Results


def run_ivae(args, config, method="ivae"):
    wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    n_layer = config.n_layer
    dataset = config.dataset
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = [args.n_obs_per_seg] if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch

    epoch_interval = 100 # save result every n epochs
    res_corr = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_recovered_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_ground_truth_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_metric = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}

    for l in n_layer:
        for n in n_obs_per_seg:
            
            if experiment == "sim":
                # generate synthetic data
                x, y, s = generate_synthetic_data(latent_dim, n_segment, n*3, l, seed=data_seed,
                    simulationMethod=dataset, one_hot_labels=True, varyMean=False)
                
                x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n)

            elif experiment == "img":
                data = sio.loadmat(data_path)
                
                x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test  = split_img_data(data)

            print(f'Running {method} experiment with L={l}; n_obs_per_seg={n}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
            
            # intiate iVAE model for each modality
            model_ivae_list = []
            data_dim = x_train.shape[1]
            aux_dim = y_train.shape[1]

            for m in range(n_modality):
                ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_seed{seed}_modality{m+1}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
                
                model_ivae = iVAE(latent_dim, 
                                data_dim, 
                                aux_dim, 
                                activation='lrelu', 
                                device=device, 
                                n_layer=l, 
                                hidden_dim=latent_dim * 2,
                                method=method,
                                seed=seed)
                
                for e in range(n_epoch):
                    print(f'Epoch: {e}')
                    [res_ivae_train, res_ivae_valid], model_ivae, params_ivae = IVAE_wrapper(X=x_train[:,:,m], U=y_train, batch_size=batch_size_ivae, 
                                                                                            n_layer=l, hidden_dim=latent_dim * 2, cuda=cuda, max_iter=mi_ivae, 
                                                                                            lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, test=False, 
                                                                                            model=model_ivae, Xv=x_valid[:,:,m], Uv=y_valid)
                    
                    if e % epoch_interval == 0:
                        print('\n')
                        res_ivae, _, _ = IVAE_wrapper(X=x_test[:,:,m], U=y_test, batch_size=batch_size_ivae, n_layer=l, hidden_dim=latent_dim * 2, 
                                                    cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, test=True, model=model_ivae)
                        
                        res_ivae = res_ivae.detach().numpy()
                        res_ivae_train = res_ivae_train.detach().numpy()
                        res_ivae_valid = res_ivae_valid.detach().numpy()
                        res_recovered_source[l][n][e].append({'test': res_ivae, 'train': res_ivae_train, 'valid': res_ivae_valid})

                        if experiment == 'sim':
                            res_corr[l][n][e].append(mean_corr_coef(res_ivae, s_test[:,:,m]))
                            res_corr[l][n][e].append(mean_corr_coef_per_segment(res_ivae, s_test[:,:,m], y_test))
                            print(res_corr[l][n][e][0])
                            
                            if m == n_modality - 1: # last epoch, last modality
                                res_ground_truth_source[l][n][e].append({'test': s_test, 'train': s_train, 'valid': s_valid})
                                res_ivae_stack = np.dstack([r['test'] for r in res_recovered_source[l][n][e]])
                                metric = MMSE(res_ivae_stack, s_test, y_test)
                                res_metric[l][n][e].append(metric)

                                res = {
                                    'mcc': res_corr[l][n][e],
                                    'recovered_source': res_recovered_source[l][n][e],
                                    'ground_truth_source': res_ground_truth_source[l][n][e],
                                    'metric': res_metric[l][n][e]
                                }

                                fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_epoch{e}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                pickle.dump(res, open(fname, "wb"))
                        
                        elif experiment == 'img':
                            if m == n_modality - 1:
                                res = {
                                    'recovered_source': res_recovered_source[l][n][e]
                                }

                                fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_epoch{e}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                pickle.dump(res, open(fname, "wb"))
                            
                model_ivae_list.append(model_ivae)

    # prepare output
    Results = {
        'mcc': res_corr,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'metric': res_metric
    }

    return Results


def run_jivae(args, config, method="jivae"):
    wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    n_layer = config.n_layer
    dataset = config.dataset
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = [args.n_obs_per_seg] if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch

    epoch_interval = 100 # save result every n epochs
    res_corr = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_recovered_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_ground_truth_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_metric = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}

    for l in n_layer:
        for n in n_obs_per_seg:
            
            if experiment == "sim":
                # generate synthetic data
                x, y, s = generate_synthetic_data(latent_dim, n_segment, n*3, l, seed=data_seed,
                    simulationMethod=dataset, one_hot_labels=True, varyMean=False)
                
                x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n)

            elif experiment == "img":
                data = sio.loadmat(data_path)
                
                x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test = split_img_data(data)

            print(f'Running {method} experiment with L={l}; n_obs_per_seg={n}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
            
            # run a single iVAE model on concatenated modalities along sample dimension
            ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_seed{seed}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
            x_concat_train = np.vstack([x_train[:,:,m] for m in range(n_modality)])
            y_concat_train = np.concatenate([y_train]*n_modality, axis=0)
            x_concat_valid = np.vstack([x_valid[:,:,m] for m in range(n_modality)])
            y_concat_valid = np.concatenate([y_valid]*n_modality, axis=0)
            x_concat_test = np.vstack([x_test[:,:,m] for m in range(n_modality)])
            y_concat_test = np.concatenate([y_test]*n_modality, axis=0)

            data_dim = x_concat_train.shape[1]
            aux_dim = y_concat_train.shape[1]
            n_sample_train = x_train.shape[0]
            n_sample_valid = x_valid.shape[0]
            n_sample_test = x_test.shape[0]
            
            model_ivae = iVAE(latent_dim, 
                            data_dim, 
                            aux_dim, 
                            activation='lrelu', 
                            device=device, 
                            n_layer=l, 
                            hidden_dim=latent_dim * 2,
                            method=method,
                            seed=seed)
            
            for e in range(n_epoch):
                print(f'Epoch: {e}')
                [res_ivae_train, res_ivae_valid], model_ivae, params_ivae = IVAE_wrapper(X=x_concat_train, U=y_concat_train, batch_size=batch_size_ivae, 
                                                                                        n_layer=l, hidden_dim=latent_dim * 2, cuda=cuda, max_iter=mi_ivae, 
                                                                                        lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, test=False, 
                                                                                        model=model_ivae, Xv=x_concat_valid, Uv=y_concat_valid)
                
                if e % epoch_interval == 0:
                    print('\n')
                    res_ivae, _, _ = IVAE_wrapper(X=x_concat_test, U=y_concat_test, batch_size=batch_size_ivae, n_layer=l, hidden_dim=latent_dim * 2, 
                                                cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, test=True, model=model_ivae)
                    
                    res_ivae = res_ivae.detach().numpy()
                    res_ivae_train = res_ivae_train.detach().numpy()
                    res_ivae_valid = res_ivae_valid.detach().numpy()

                    for m in range(n_modality):
                        res_recovered_source[l][n][e].append({'train':res_ivae_train[m*n_sample_train:(m+1)*n_sample_train,:], \
                                                              'valid':res_ivae_valid[m*n_sample_valid:(m+1)*n_sample_valid,:], \
                                                              'test':res_ivae[m*n_sample_test:(m+1)*n_sample_test,:]})
                        
                    if experiment == 'sim':
                        for m in range(n_modality):
                            res_corr[l][n][e].append(mean_corr_coef(res_ivae[m*n_sample_test:(m+1)*n_sample_test,:], s_test[:,:,m]))
                            res_corr[l][n][e].append(mean_corr_coef_per_segment(res_ivae[m*n_sample_test:(m+1)*n_sample_test,:], s_test[:,:,m], y_test))
                            print(res_corr[l][n][e][0])
                        
                        res_ground_truth_source[l][n][e].append({'test':s_test, 'train':s_train, 'valid':s_valid})
                        res_ivae_stack = np.dstack([r['test'] for r in res_recovered_source[l][n][e]])
                        metric = MMSE(res_ivae_stack, s_test, y_test)
                        res_metric[l][n][e].append(metric)

                        res = {
                            'mcc': res_corr[l][n][e],
                            'recovered_source': res_recovered_source[l][n][e],
                            'ground_truth_source': res_ground_truth_source[l][n][e],
                            'metric': res_metric[l][n][e]
                        }

                        fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_epoch{e}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                        pickle.dump(res, open(fname, "wb"))
                    
                    elif experiment == 'img':
                        res = {
                            'recovered_source': res_recovered_source[l][n][e]
                        }

                        fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_epoch{e}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                        pickle.dump(res, open(fname, "wb"))

    # prepare output
    Results = {
        'mcc': res_corr,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'metric': res_metric
    }

    return Results


def run_givae(args, config, method="givae"):
    wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    n_layer = config.n_layer
    dataset = config.dataset
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = [args.n_obs_per_seg] if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch

    epoch_interval = 100 # save result every n epochs
    res_corr = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_recovered_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_ground_truth_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_metric = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}

    for l in n_layer:
        for n in n_obs_per_seg:
            
            if experiment == "sim":
                # generate synthetic data
                x, y, s = generate_synthetic_data(latent_dim, n_segment, n*3, l, seed=data_seed,
                    simulationMethod=dataset, one_hot_labels=True, varyMean=False)
                
                x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n)

            elif experiment == "img":
                data = sio.loadmat(data_path)
                
                x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test = split_img_data(data)

            print(f'Running {method} experiment with L={l}; n_obs_per_seg={n}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
            
            # run a single iVAE model on concatenated modalities along feature dimension
            ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_givae_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_seed{seed}_epoch{n_epoch}_maxiter{mi_ivae}_lrivae{lr_ivae}.pt')
            x_concat_train = np.hstack([x_train[:,:,m] for m in range(n_modality)])
            x_concat_valid = np.hstack([x_valid[:,:,m] for m in range(n_modality)])
            x_concat_test = np.hstack([x_test[:,:,m] for m in range(n_modality)])
            data_dim = x_concat_train.shape[1]
            aux_dim = y_train.shape[1]
            
            model_ivae = iVAE(latent_dim, 
                            data_dim, 
                            aux_dim, 
                            activation='lrelu', 
                            device=device, 
                            n_layer=l, 
                            hidden_dim=latent_dim * 2,
                            method=method,
                            seed=seed)
            
            for e in range(n_epoch):
                print(f'Epoch: {e}')
                [res_ivae_train, res_ivae_valid], model_ivae, params_ivae = IVAE_wrapper(X=x_concat_train, U=y_train, batch_size=batch_size_ivae, n_layer=l, 
                                                                                        hidden_dim=latent_dim * 2, cuda=cuda, max_iter=mi_ivae, lr=lr_ivae,
                                                                                        ckpt_file=ckpt_file, seed=seed, test=False, model=model_ivae,
                                                                                        Xv=x_concat_valid, Uv=y_valid)
                
                if e % epoch_interval == 0:
                    print('\n')
                    res_ivae, _, _ = IVAE_wrapper(X=x_concat_test, U=y_test, batch_size=batch_size_ivae, n_layer=l, hidden_dim=latent_dim * 2, 
                                                cuda=cuda, max_iter=mi_ivae, lr=lr_ivae, ckpt_file=ckpt_file, seed=seed, test=True, model=model_ivae)
                    
                    res_ivae = res_ivae.detach().numpy()
                    res_ivae_train = res_ivae_train.detach().numpy()
                    res_ivae_valid = res_ivae_valid.detach().numpy()
                    for m in range(n_modality):
                        res_recovered_source[l][n][e].append({'test':res_ivae, 'train':res_ivae_train, 'valid':res_ivae_valid})

                    if experiment == 'sim':
                        for m in range(n_modality):
                            res_corr[l][n][e].append(mean_corr_coef(res_ivae, s_test[:,:,m]))
                            res_corr[l][n][e].append(mean_corr_coef_per_segment(res_ivae, s_test[:,:,m], y_test))
                            print(res_corr[l][n][e][0])
                        
                        res_ground_truth_source[l][n][e].append({'test':s_test, 'train':s_train, 'valid':s_valid})
                        res_ivae_stack = np.dstack([r['test'] for r in res_recovered_source[l][n][e]])
                        metric = MMSE(res_ivae_stack, s_test, y_test)
                        res_metric[l][n][e].append(metric)

                        res = {
                            'mcc': res_corr[l][n][e],
                            'recovered_source': res_recovered_source[l][n][e],
                            'ground_truth_source': res_ground_truth_source[l][n][e],
                            'metric': res_metric[l][n][e]
                        }

                        fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_epoch{e}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                        pickle.dump(res, open(fname, "wb"))
                    
                    elif experiment == 'img':                        
                        res = {
                            'recovered_source': res_recovered_source[l][n][e]
                        }

                        fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_epoch{e}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                        pickle.dump(res, open(fname, "wb"))

    # prepare output
    Results = {
        'mcc': res_corr,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'metric': res_metric
    }

    return Results


def run_misa_ivae(args, config, method="misa"):
    wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    index = slice(0, n_modality)
    experiment = config.experiment
    n_layer = config.n_layer
    dataset = config.dataset
    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    cuda = config.cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # iVAE config
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = [args.n_obs_per_seg] if args.n_obs_per_seg else config.n_obs_per_seg
    batch_size_ivae = args.ivae_batch_size if args.ivae_batch_size else config.ivae.batch_size
    lr_ivae = args.ivae_lr if args.ivae_lr else config.ivae.lr
    mi_ivae = args.ivae_max_iter_per_epoch if args.ivae_max_iter_per_epoch else config.ivae.max_iter_per_epoch

    # MISA config
    input_dim = [latent_dim] * n_modality
    output_dim = [latent_dim] * n_modality
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
    
    if len(beta) > 0:
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
        if len(beta) == 1:
            beta = beta*torch.ones(subspace[0].size(-2), device=device)
    
    if len(lam) > 0:
        lam = torch.tensor(lam, dtype=torch.float32, device=device)
        if len(lam) == 1:
            lam = lam*torch.ones(subspace[0].size(-2), device=device)
    
    batch_size_misa = args.misa_batch_size if args.misa_batch_size else config.misa.batch_size

    epoch_interval = 100 # save result every n epochs
    res_corr = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_recovered_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_ground_truth_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_metric = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}

    for l in n_layer:
        for n in n_obs_per_seg:
            
            if experiment == "sim":
                # generate synthetic data
                x, y, s = generate_synthetic_data(latent_dim, n_segment, n*3, l, seed=data_seed,
                    simulationMethod=dataset, one_hot_labels=True, varyMean=False)
                
                x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n)

                initial_weights = []
            
            elif experiment == "img":
                data = sio.loadmat(data_path)
                
                x_train, y_train, u_train, x_valid, y_valid, u_valid, x_test, y_test, u_test = split_img_data(data)

                initial_weights = []
                # initial_weights = [ np.eye(data_dim) for _ in range(n_modality) ]
            
            lr_misa = lr_ivae/n_segment
            mi_misa = mi_ivae
            
            print(f'Running {method} experiment with L={l}; n_obs_per_seg={n}; n_seg={n_segment}; n_source={latent_dim}; seed={seed}; n_epoch={n_epoch}; max_iter_per_epoch={mi_ivae}; lr_ivae={lr_ivae}')
            
            loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}

            ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_seed{seed}_epoch{n_epoch}_maxiter{mi_misa}_lrmisa{round(lr_misa, 5)}.pt')
            
            # initialize MISA model weights using iVAE sources as A = (z^T z)^{-1} z^T X
            for m in range(n_modality):
                fname = os.path.join(args.run, f'src_ivae_m{m+1}_{experiment}_diva_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                if os.path.exists(fname):
                    print(f'Loading iVAE source {fname}')
                    res_ivae = pickle.load(open(fname, 'rb'))
                    weight_init = np.linalg.inv(res_ivae.T @ res_ivae) @ res_ivae.T @ x_train[:,:,m]
                    initial_weights.append(weight_init)

            model_misa = MISA(weights=initial_weights,
                index=index, 
                subspace=subspace, 
                eta=eta, 
                beta=beta, 
                lam=lam, 
                input_dim=input_dim, 
                output_dim=output_dim, 
                seed=seed, 
                device=device,
                latent_dim=latent_dim)

            # update iVAE and MISA model weights
            # run iVAE per modality
            np.random.seed(seed)
            segment_shuffled = np.arange(n_segment)
            np.random.shuffle(segment_shuffled)

            if experiment == "sim":
                res_train = np.zeros_like(s_train)
                res_valid = np.zeros_like(s_valid)
                res_test = np.zeros_like(s_test)
            else:
                res_train = np.zeros((x_train.shape[0], latent_dim, n_modality))
                res_valid = np.zeros((x_valid.shape[0], latent_dim, n_modality))
                res_test = np.zeros((x_test.shape[0], latent_dim, n_modality))

            for e in range(n_epoch):
                print(f'Epoch: {e}')
                # loop MISA through segments
                # remove the mean of segment because MISA loss assumes zero mean
                # randomize segment order
                for seg in segment_shuffled:
                    if experiment == "sim":
                        x_seg_train = x_train[seg*n:(seg+1)*n,:,:]
                        x_seg_valid = x_valid[seg*n:(seg+1)*n,:,:]
                        x_seg_test = x_test[seg*n:(seg+1)*n,:,:]
                    elif experiment == "img":
                        ind_train = np.where(u_train[:,seg]==1)[0]
                        x_seg_train = x_train[ind_train,:,:]
                        ind_valid = np.where(u_valid[:,seg]==1)[0]
                        x_seg_valid = x_valid[ind_valid,:,:]
                        ind_test = np.where(u_test[:,seg]==1)[0]
                        x_seg_test = x_test[ind_test,:,:]
                    
                    # remove mean of segment
                    x_seg_dm_train = x_seg_train - np.mean(x_seg_train, axis=0)
                    x_seg_dm_valid = x_seg_valid - np.mean(x_seg_valid, axis=0)
                    x_seg_dm_test = x_seg_test - np.mean(x_seg_test, axis=0)

                    # a list of datasets, each dataset dimension is sample x source
                    ds_train = Dataset(data_in=x_seg_dm_train, device=device)
                    ds_valid = Dataset(data_in=x_seg_dm_valid, device=device)
                    ds_test = Dataset(data_in=x_seg_dm_test, device=device)
                    data_loader_train = DataLoader(dataset=ds_train, batch_size=len(ds_train), shuffle=False)
                    data_loader_valid = DataLoader(dataset=ds_valid, batch_size=len(ds_valid), shuffle=False)
                    data_loader_test = DataLoader(dataset=ds_test, batch_size=len(ds_test), shuffle=False)

                    model_misa, [sr_train, sr_valid], _ = MISA_wrapper_(data_loader=data_loader_train,
                                        test_data_loader=data_loader_valid,
                                        epochs=mi_misa,
                                        lr=lr_misa,
                                        device=device,
                                        ckpt_file=ckpt_file,
                                        model_MISA=model_misa)

                    _, sr_test, _ = MISA_wrapper_(test=True, data_loader=data_loader_test, device=device, 
                                                  ckpt_file=ckpt_file, model_MISA=model_misa)
                    
                    if e % epoch_interval == 0:
                        for m in range(n_modality):
                            if experiment == "sim":
                                res_train[seg*n:(seg+1)*n,:,m] = sr_train[m].detach().numpy()
                                res_valid[seg*n:(seg+1)*n,:,m] = sr_valid[m].detach().numpy()
                                res_test[seg*n:(seg+1)*n,:,m] = sr_test[m].detach().numpy()
                            elif experiment == "img":
                                res_train[ind_train,:,m] = sr_train[m].detach().numpy()
                                res_valid[ind_valid,:,m] = sr_valid[m].detach().numpy()
                                res_test[ind_test,:,m] = sr_test[m].detach().numpy()
                
                if e % epoch_interval == 0:
                    for m in range(n_modality):
                        
                        res_recovered_source[l][n][e].append({'train': res_train[:,:,m], 'valid': res_valid[:,:,m], 'test': res_test[:,:,m]})

                        if experiment == 'sim':
                            res_corr[l][n][e].append(mean_corr_coef(res_test[:,:,m], s_test[:,:,m]))
                            res_corr[l][n][e].append(mean_corr_coef_per_segment(res_test[:,:,m], s_test[:,:,m], y_test))
                            print(res_corr[l][n][e][0])
                            
                            if m == n_modality - 1: # last epoch, last modality
                                res_ground_truth_source[l][n][e].append({'train': s_train, 'valid': s_valid, 'test': s_test})
                                metric = MMSE(res_test, s_test, y_test)
                                res_metric[l][n][e].append(metric)

                                res = {
                                    'mcc': res_corr[l][n][e],
                                    'recovered_source': res_recovered_source[l][n][e],
                                    'ground_truth_source': res_ground_truth_source[l][n][e],
                                    'metric': res_metric[l][n][e]
                                }

                                fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                pickle.dump(res, open(fname, "wb"))

                        elif experiment == 'img':
                            if m == n_modality - 1:
                                res = {
                                    'recovered_source': res_recovered_source[l][n][e]
                                }

                                fname = os.path.join(args.run, f'res_{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_epoch{e}_bsmisa{batch_size_misa}_bsivae{batch_size_ivae}_lrivae{lr_ivae}_maxiter{mi_ivae}_seed{seed}.p')
                                pickle.dump(res, open(fname, "wb"))

    # prepare output
    Results = {
        'mcc': res_corr,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'metric': res_metric
    }

    return Results


def run_jicebeem(args, config, method="jicebeem"):
    wandb.init(project=method, entity="deepmisa")

    seed = args.seed
    data_seed = config.data_seed
    data_path = args.data_path
    n_modality = config.n_modality
    experiment = config.experiment
    n_layer = config.n_layer
    cuda = config.cuda
    device = config.device
    dataset = config.dataset

    n_epoch = args.n_epoch if args.n_epoch else config.n_epoch
    latent_dim = args.n_source if args.n_source else config.latent_dim
    n_segment = args.n_segment if args.n_segment else config.n_segment
    n_obs_per_seg = [args.n_obs_per_seg] if args.n_obs_per_seg else config.n_obs_per_seg
    
    lr_flow = config.icebeem.lr_flow
    lr_ebm = config.icebeem.lr_ebm
    n_layer_flow = config.icebeem.n_layer_flow
    ebm_hidden_size = config.icebeem.ebm_hidden_size

    epoch_interval = 100 # save result every n epochs
    res_corr = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_recovered_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_ground_truth_source = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}
    res_metric = {l: {n: {e: [] for e in [n*epoch_interval for n in range(n_epoch//epoch_interval+1) ]} for n in n_obs_per_seg} for l in n_layer}

    for l in n_layer:
        for n in n_obs_per_seg:
            
            if experiment == "sim":
                # generate synthetic data
                x, y, s = generate_synthetic_data(latent_dim, n_segment, n*3, l, seed=data_seed,
                    simulationMethod=dataset, one_hot_labels=True, varyMean=False)
                
                x_train, y_train, s_train, x_valid, y_valid, s_valid, x_test, y_test, s_test = split_sim_data(x, y, s, n_segment, n)

            elif experiment == "img":
                data = sio.loadmat(data_path)
                
                x_train, y_train, x_valid, y_valid, x_test, y_test = split_img_data(data)

            n_layer_ebm = l + 1
            ckpt_file = os.path.join(args.run, 'checkpoints', f'{experiment}_{method}_layer{l}_source{latent_dim}_obs{n}_seg{n_segment}_seed{seed}_epoch{n_epoch}_layerebm{n_layer_ebm}_layerflow{n_layer_flow}_lrebm{lr_ebm}_lrflow{lr_flow}.pt')
            x_concat = np.vstack([x_train[:,:,m] for m in range(n_modality)])
            y_concat = np.concatenate([y_train]*n_modality, axis=0)
            s_concat = np.vstack([s_train[:,:,m] for m in range(n_modality)])
            n_sample = s_train.shape[0]

            for e in range(n_epoch):
                print(f'Epoch: {e}')
                recovered_source_list = ICEBEEM_wrapper(X=x_concat, Y=y_concat, ebm_hidden_size=ebm_hidden_size,
                                                n_layer_ebm=n_layer_ebm, n_layer_flow=n_layer_flow,
                                                lr_flow=lr_flow, lr_ebm=lr_ebm, seed=seed, ckpt_file=ckpt_file,
                                                test=False)
            
                if e % epoch_interval == 0:
                    mcc_list = [mean_corr_coef_per_segment(z, s_concat, y_concat)[0][0] for z in recovered_source_list]
                    ind = np.argmax(mcc_list)
                    recovered_source = recovered_source_list[ind]

                    for m in range(n_modality):
                        res_corr[l][n][e].append(mean_corr_coef(recovered_source[m*n_sample:(m+1)*n_sample,:], s_train[:,:,m]))
                        res_corr[l][n][e].append(mean_corr_coef_per_segment(recovered_source[m*n_sample:(m+1)*n_sample,:], s_train[:,:,m], y_train))
                        print(res_corr[l][n][e][0])
                        res_recovered_source[l][n][e].append(recovered_source[m*n_sample:(m+1)*n_sample,:])
                        
                    res_ground_truth_source[l][n][e].append(s_train)
                    recovered_source = np.dstack(res_recovered_source[l][n][e])
                    metric = MMSE(recovered_source, s_train, y_train)
                    res_metric[l][n][e].append(metric)

                    res = {
                        'mcc': res_corr[l][n][e],
                        'recovered_source': res_recovered_source[l][n][e],
                        'ground_truth_source': res_ground_truth_source[l][n][e],
                        'metric': res_metric[l][n][e]
                    }

    # prepare output
    Results = {
        'mcc': res_corr,
        'recovered_source': res_recovered_source,
        'ground_truth_source': res_ground_truth_source,
        'metric': res_metric
    }

    return Results
