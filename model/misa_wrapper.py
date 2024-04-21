import torch
import wandb
import numpy as np
from model.MISAK import MISA

def MISA_wrapper_(data_loader, epochs=10, lr=0.01, A=None, device='cpu', ckpt_file='misa.pt', test=False, test_data_loader=None, model_MISA=None):
    
    model_MISA.to(device=device)
    
    final_MISI = []
    
    if not test:
        training_loss, training_MISI, optimizer = model_MISA.train_me(data_loader, epochs, lr, A)
        if len(training_MISI) > 0:
            final_MISI = training_MISI[-1]
        training_loss_last_iter_avg = np.mean(training_loss[-1])
        training_output = model_MISA.output

        res_dict = {'model_MISA': model_MISA.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'seed': model_MISA.seed,
                    'index': model_MISA.index,
                    'subspace': model_MISA.subspace,
                    'eta': model_MISA.eta,
                    'beta': model_MISA.beta,
                    'lam': model_MISA.lam,
                    'training_loss': training_loss,
                    'training_output': training_output,
                    'training_MISI': training_MISI}
        
        if test_data_loader is not None:
            validation_loss = model_MISA.predict(test_data_loader)
            validation_loss_avg = np.mean(validation_loss)
            validation_output = model_MISA.output
            res_dict['validation_loss'] = validation_loss
            res_dict['validation_output'] = validation_output
            output_MISA = [training_output, validation_output]
            print(f"MISA training loss: {training_loss_last_iter_avg:.3f}; validation loss: {validation_loss_avg:.3f}")
            wandb.log({'MISA training loss': training_loss_last_iter_avg, 'MISA validation loss': validation_loss_avg})
        else:
            output_MISA = training_output
            print(f"MISA training loss: {training_loss_last_iter_avg:.3f}")
            wandb.log({'MISA training loss': training_loss_last_iter_avg})
        
        torch.save(res_dict, ckpt_file)
        # print("Saved checkpoint to: " + ckpt_file)

    else:
        checkpoint = torch.load(ckpt_file)
        model_MISA.load_state_dict(checkpoint['model_MISA'])
        test_loss = model_MISA.predict(data_loader)
        test_loss_avg = np.mean(test_loss)
        test_output = model_MISA.output
        output_MISA = test_output
        print(f"MISA test loss: {test_loss_avg:.3f}")
    
    return model_MISA, output_MISA, final_MISI


def MISA_wrapper(data_loader, index, subspace, eta, beta, lam, input_dim, output_dim, seed, epochs, lr,
                 weights=list(), A=None, device='cpu', ckpt_file='misa.pt', test=False, test_data_loader=None, model=None):
    
    model_MISA=MISA(weights=weights,
                 index=index, 
                 subspace=subspace, 
                 eta=eta, 
                 beta=beta, 
                 lam=lam, 
                 input_dim=input_dim, 
                 output_dim=output_dim,
                 seed=seed,
                 device=device,
                 model=model)
    
    model_MISA.to(device=device)
    
    final_MISI = []
    
    if not test:
        training_loss, training_MISI, optimizer = model_MISA.train_me(data_loader, epochs, lr, A)
        if len(training_MISI) > 0:
            final_MISI = training_MISI[-1]
        
        test_loss = model_MISA.predict(test_data_loader)
        print(f"test loss: {test_loss[0].detach().cpu().numpy():.3f}")

        torch.save({'model_MISA': model_MISA.state_dict(),
                'optimizer': optimizer.state_dict(),
                'seed': model_MISA.seed,
                'index': model_MISA.index,
                'subspace': model_MISA.subspace,
                'eta': model_MISA.eta,
                'beta': model_MISA.beta,
                'lam': model_MISA.lam,
                'training_loss': training_loss,
                'training_MISI': training_MISI,
                'test_loss': test_loss},
               ckpt_file)
        print("Saved checkpoint to: " + ckpt_file)

    else:
        checkpoint = torch.load(ckpt_file)
        model_MISA.load_state_dict(checkpoint['model_MISA'])
        test_loss = model_MISA.predict(test_data_loader)
        print(f"test loss: {test_loss[0].detach().cpu().numpy():.3f}")
    
    return model_MISA, final_MISI