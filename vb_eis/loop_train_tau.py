import argparse
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints

from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import Adam
from tqdm import tqdm, trange

import tau_models

def per_param_callable_old(module_name, param_name):
    return {"lr": 0.01}
    if param_name == 'N_par':
        return {"lr": 0.001}
    else:
        return {"lr": 0.01}

def per_param_callable(module_name, param_name):
    # return {"lr": 0.01}
    if param_name == 'N_par':
        return {"lr": 0.01,'betas': (0.95, 0.999)}
    if param_name.startswith('R_'):
        return {"lr": 0.1,'betas': (0.95, 0.999)}
    else:
        return {"lr": 0.01,'betas': (0.95, 0.999)}

def train(Z, F_all, its, max_dim, num_particles, out_prefix, grace=1500, 
        annealing_epochs=500, R_prior = torch.tensor(1e-3), Rs_prior = torch.tensor(1e-3), dir_alpha=1.):
    device = 'cuda'
    exp_data = torch.cat((torch.tensor(Z.imag), torch.tensor(Z.real)),0).float().unsqueeze(-1)
    s = torch.tensor(2*np.pi*F_all.astype(np.float32)*1j).unsqueeze(-1).to(device)
    train_data = exp_data.t().float().to(device)

    
    # enum_model = Enumerated_Basic_no_poles(max_dim, s, device=device, Rs_prior=torch.tensor(1e-3), R_prior=torch.tensor(1e-3))
    enum_model = tau_models.Enumerated_Basic_with_mask(max_dim, s, device=device, R_prior = R_prior, Rs_prior = Rs_prior, sigma=torch.tensor(1e-6),dir_alpha=dir_alpha)
    # enum_mode = fos_models.LukaOld(s)
    model = enum_model.model
    guide = enum_model.guide

    annealing_factor = np.tanh(np.linspace(-np.pi, np.pi,grace + annealing_epochs - grace)) / 2 +0.5 + np.finfo(np.float32).eps
    annealing_factor = np.pad(annealing_factor,(grace, its - (grace + annealing_epochs)),'constant', constant_values=(np.finfo(np.float32).eps, 1))
    annealing_factor = np.clip(annealing_factor,0,1)

    pyro.enable_validation(True)
    pyro.clear_param_store() #always clear the stored parameters before running a new session of SVI

    optimizer = Adam(per_param_callable)

    #Define our SVI with the parameters for model,guide,optimizer and the type of loss function we want to use.
    #For our example we have Adam optimizer with learning rate as set before and the loss function is Trace_ELBO.
    # Register hooks to monitor gradient norms. Since we want to monitor gradient norms and losses to monitor convergence.
    svi = SVI(model,
            guide,
            # Adam({"lr": lr}),
            optim=optimizer,
            # loss=Trace_ELBO())
               loss=TraceEnum_ELBO(max_plate_nesting=1,num_particles=num_particles,vectorize_particles=True))

    eloss = []
    N_progress = []
    i=0
    # pbar = trange(its)
    for i in trange(its):
    # while((i<its )):
        elbo = svi.step(train_data, annealing_factor = annealing_factor[i])#only parameter for this function is the data
        eloss.append(elbo)   #store elbo values for progression overview
        N_progress.append(pyro.param("N_par").cpu().detach().numpy()) #store pole values for progression overview

        # if i % 100 == 0:
        #     pbar.set_description(f"elbo is: {np.log10(elbo):.2f}")
            # print(str(i)+"-th iteration, elbo is: " + str(elbo) +'\n')
        # i+=1

    
    torch.save({'N_progress':N_progress,'eloss':eloss}, f'{out_prefix}_model.pt')
    xx = pyro.get_param_store().get_state()#.save(f'{out_prefix}_model_pyro.pt')
    for k,v in xx['params'].items():
        xx['params'][k] = v.to('cpu')
    torch.save(xx, f'{out_prefix}_model_pyro.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--N', help='Index', required=True, type=int)
    parser.add_argument('--epochs', help='Number of epochs', required=True, type=int)
    parser.add_argument('--num_particles', help='Number of particles', required=True, type=int)

    parser.add_argument("--Z", help="NPZ file with Z", required=False, type=str)
    parser.add_argument("--simulated", help="Flag for simulated data", required=False, type=bool, default=False)
    parser.add_argument("--out_prefix", help="Prefix of the output file", required=True, type=str)
    parser.add_argument("--max_dim", help="Max number of poles", required=True, type=int)
    parser.add_argument("--annealing_epochs", help="Number of annealing epochs", required=False, type=int, default=0)
    parser.add_argument("--grace", help="Number of grace epochs", required=False, type=int, default=0)
    parser.add_argument("--Rs_prior", help="Prior for Rs", required=False, type=float, default=1e-3)
    parser.add_argument("--R_prior", help="Prior for R", required=False, type=float, default=1e-3)
    parser.add_argument("--data_path", help="Path for datafiles", required=False, type=str, default='./')
    parser.add_argument("--alpha", help="Dirichlet concentration parameter", required=False, type=float, default=1.0)
    

    args = parser.parse_args()

    if not args.Z and not args.simulated:
        parser.error("Either --Z or --simulated must be set")
    elif args.Z and args.simulated:
        parser.error("Only one of --Z or --simulated can be set")

    # Check if annealing_epochs + grace < epochs. If so scale the annealing_epochs and grace to fit epochs
    if args.annealing_epochs + args.grace > args.epochs:
        print("Number of annealing_epochs + grace is larger than epochs. Scaling annealing_epochs and grace")
        args.annealing_epochs = int(args.epochs * args.annealing_epochs / (args.annealing_epochs + args.grace))
        args.grace = args.epochs - args.annealing_epochs

        print(f'New annealing_epochs: {args.annealing_epochs}')
        print(f'New grace: {args.grace}')

    if args.simulated:
        data = np.load(f"{args.data_path}numerical_sim_{args.N}.npz",allow_pickle=True)
        Z_all = data['mat_Z'].flatten()[::2]
        F_all = data['f'].flatten()[::2]
    else:
        data = np.load(args.Z,allow_pickle=True)
        Z_all = data['mat_Z'][args.N].flatten()
        F_all = data['f_all'].flatten()

    

    train(Z_all, F_all, args.epochs, args.max_dim, args.num_particles, f'{args.out_prefix}_{args.N}',
            R_prior=torch.tensor(args.R_prior), Rs_prior=torch.tensor(args.Rs_prior), annealing_epochs=args.annealing_epochs,grace=args.grace, dir_alpha=args.alpha)
