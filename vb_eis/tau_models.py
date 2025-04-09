import torch
from torch.distributions import constraints
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.infer import config_enumerate

def EIS_curve_3RQ_recursive(s,Rs,R,Q,alfa,L,Pole):
    Z = Rs + s*L
    for i in range (0,Pole):
        rq = 1+torch.pow(s*R[i]*Q[i],alfa[i])
        Z += R[i]*torch.pow(rq,-1)
    
    # return Z
    # y=torch.cat((Im_impe.to(torch.float32), Re_impe.to(torch.float32)),0)
    y = torch.cat((Z.imag, Z.real),0)
    y = y.t()
    return y.squeeze()

def cot(theta):
    return 1/torch.tan(theta)

def csc(theta):
    return 1/torch.sin(theta)



def pole_importance_calc(R, tau, alfa):
    lim0 = R * torch.atan(cot(torch.pi * alfa ))
    lim_inf = R * torch.abs( csc( torch.pi * alfa ) ) * torch.sin(torch.pi * alfa )/(2*alfa )
    return lim_inf - lim0

def order_poles(R, tau, alfa):
    pole_importance = pole_importance_calc(R, tau, alfa)
    return torch.argsort(pole_importance,descending=True)

def generate_pole_mask(R, tau, alfa, pole):
    ind = order_poles(R, tau, alfa)
    max_idx = pole.max().item() + 1
    pole_mask = torch.cumsum(torch.nn.functional.one_hot(ind[:max_idx], num_classes=len(ind)), dim=0)[pole,:]
    return pole_mask

def vector_parallel_EIS_curve_tau_recursive(s, Rs, R, tau, alfa, L, Pole, device='cpu', ret_mask=False):
    """Calculates the EIS curve for a R tau circuit using a recursive algorithm.
    
    Args:
    s: A tensor of complex frequencies.
    Rs: A tensor of solution resistances.
    R: A tensor of charge transfer resistances.
    tau: A tensor of time constants.
    alfa: A tensor of Warburg exponents.
    L: A tensor of inductances.
    Pole: A tensor of poles.
    device: The device to use for the calculation.
    
    Returns:
    A tensor of complex impedances.
    """

    Pole = Pole.flatten()
    a_all = torch.pow(s*torch.pow(10,tau.unsqueeze(-2)), alfa.unsqueeze(-2))
    Z = Rs.unsqueeze(-1) + s*L
    Z_poles = R.unsqueeze(-2)*torch.pow(1+a_all,-1)
    N = Z_poles.shape[-1]
    rows = torch.arange(N).unsqueeze(1).to(device)


    if Pole.dim() < 2:
        pole_mask = rows <= Pole
    else:
        pole_mask = rows <= Pole.permute(*torch.arange(Pole.ndim - 1, -1, -1))

    indicator_matrix = pole_mask.float() + 1j*0

    ind = order_poles(R, tau, alfa)
    max_idx = Pole.max().item() + 1

    # Use cumulative sum to build a mask for the indices in 'ind' up to each index in 'poles'
    pole_mask = torch.cumsum(torch.nn.functional.one_hot(ind[:max_idx], num_classes=len(ind)), dim=0)[Pole,:].T
    indicator_matrix = pole_mask.float() + 1j*0

    Z_poles = Z_poles @ indicator_matrix
    Z = Z_poles + Z
    
    
    y = torch.cat((Z.imag, Z.real),-2)
    perm_ind = torch.arange(y.ndim).roll(1)
    if ret_mask:
        return y.permute(*perm_ind), ind, pole_mask
    return y.permute(*perm_ind), ind

class Enumerated_Basic_with_mask:
    def __init__(self, max_dim, s, sim_model=vector_parallel_EIS_curve_tau_recursive, 
    Rs_prior = torch.tensor(1e-3), R_prior = torch.tensor(1e-3),  L_t = torch.tensor(1e-7),
     device='cuda', sigma = torch.tensor(1e-6),
     tau_min = None, tau_max = None, dir_alpha=1):
        self.max_dim = max_dim
        self.s = s
        self.sim_model = sim_model
        self.device = device
        self.Rs_prior = Rs_prior   
        self.R_prior = R_prior  
        self.print_debug = False
        self.sigma = sigma.to(device)
        self.L = L_t.to(device)
        f = s.cpu()*(-1j)/2/torch.pi
        f = f.real
        if tau_min is None:
            tau_min = 1/f.max()
        if tau_max is None:
            tau_max = 1/f.min()

        self.dir_alpha = dir_alpha
        self.cat_prior = dist.Dirichlet(torch.ones(self.max_dim).to(self.device)*self.dir_alpha).sample().detach()
        
        self.tau_mu_prior = torch.linspace(torch.log10(tau_min)+1, torch.log10(tau_max)-1, max_dim).to(device)
        # print(self.tau_mu_prior)
        self.ind_order = None

    @config_enumerate#(default="sequential")
    def model(self, true_y, annealing_factor=1., sim=False):
        """Defines a Pyro model for a 3RQ circuit.
        Args:
        true_y: A tensor of observed EIS data.

        Returns:
        A tensor of predicted EIS data.
        """

        # Sample the model parameters.
        Rs = pyro.sample('Rs', dist.LogNormal(torch.log(self.Rs_prior.to(self.device)),
                                          1e-1))
        with pyro.plate('dims', self.max_dim):
            R = pyro.sample('R', dist.LogNormal(torch.log(self.R_prior.expand(self.max_dim).to(self.device)),
                                                1.))
            alfa = pyro.sample('alfa', dist.Beta(torch.tensor([5.]*self.max_dim).expand(self.max_dim).to(self.device),
                                                    torch.tensor([1.]*self.max_dim).to(self.device)))
            taus = pyro.sample('tau', dist.Normal(self.tau_mu_prior,1.))
            
        with pyro.poutine.scale(None, annealing_factor):
            pole = pyro.sample('pole', dist.Categorical(self.cat_prior))

        z_mu, ind, ret_mask = self.sim_model(self.s, Rs, R, taus, alfa*0.4+0.6, self.L, pole,
                                                     self.device, ret_mask = True)

        self.ind_order = ind

        mask = ret_mask.T.bool()

        with pyro.poutine.modify(modify_attrs={'mask':mask}):
            R = pyro.sample('R', dist.LogNormal(torch.log(self.R_prior.expand(self.max_dim).to(self.device)),
                                                1.))
            alfa = pyro.sample('alfa', dist.Beta(torch.tensor([5.]*self.max_dim).to(self.device),
                                                torch.tensor([1.]*self.max_dim).to(self.device)))
            taus = pyro.sample('tau', dist.Normal(self.tau_mu_prior,1.))

        # Observe the EIS data.
        with pyro.poutine.scale(scale=1/(2 * len(self.s))):
            with pyro.plate('data', 2 * len(self.s)):
                pyro.sample('obs', dist.Normal(z_mu, self.sigma), obs=true_y)

        return z_mu, mask

    @config_enumerate#(default="sequential")
    def guide(self, true_y, annealing_factor=1., sim=False):
        N_par=pyro.param('N_par', dist.Dirichlet(torch.ones(self.max_dim).to(self.device)*self.dir_alpha),constraint=constraints.simplex)
        if annealing_factor < 1:
            N_par = N_par.detach()
        with pyro.poutine.scale(None, annealing_factor):
            pole = pyro.sample('pole', dist.Categorical(N_par))
    


        Rs_mu = pyro.param('Rs_mu', torch.log(self.Rs_prior.to(self.device)))
        Rs_sigma = pyro.param('Rs_sigma', torch.tensor(1e-1).to(self.device), constraint=constraints.positive)
        Rs = pyro.sample('Rs', dist.LogNormal(Rs_mu, Rs_sigma))
    
        # with pyro.plate('dims',max_dim) as i:
        R_mu = pyro.param('R_mu', torch.log(self.R_prior.expand(self.max_dim)).to(self.device))
        R_sigma = pyro.param('R_sigma', torch.tensor([1.]*self.max_dim).to(self.device), constraint=constraints.positive)
        
        tau_mu = pyro.param('tau_mu', self.tau_mu_prior.to(self.device), 
            # constraint=constraints.interval(self.tau_mu_prior.log().min().to(self.device), self.tau_mu_prior.log().max().to(self.device)))
                            constraint=constraints.interval(torch.tensor(-3.).to(self.device), torch.tensor(4.).to(self.device)))
        tau_sigma = pyro.param('tau_sigma', torch.tensor([1.]*self.max_dim).to(self.device), constraint=constraints.positive)    
        
        alfa_a = pyro.param('alfa_a', torch.tensor([5.]*self.max_dim).to(self.device), constraint=constraints.interval(0.5,50.0))
        alfa_b = pyro.param('alfa_b', torch.tensor([1.]*self.max_dim).to(self.device), constraint=constraints.interval(0.5,50.0))
    
        with pyro.plate('dims',self.max_dim):
            R = pyro.sample('R', dist.LogNormal(R_mu, R_sigma))
            taus = pyro.sample('tau', dist.Normal(tau_mu, tau_sigma))
            alfa = pyro.sample('alfa', dist.Beta(alfa_a, alfa_b)) 

        mask = generate_pole_mask(R, taus, alfa, pole)
        mask = mask.bool()
        with pyro.poutine.modify(modify_attrs={'mask':mask}):
            R = pyro.sample('R', dist.LogNormal(R_mu, R_sigma))
            taus = pyro.sample('tau', dist.LogNormal(tau_mu, tau_sigma))
            alfa = pyro.sample('alfa', dist.Beta(alfa_a, alfa_b)) 