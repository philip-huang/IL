import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = ["Encoder", "Bayesian_Linear", "Generator", "VAE"]

def sample_gaussian(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    sample = mu + eps*std
    return sample

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc21 = nn.Linear(500, 256)
        self.fc22 = nn.Linear(500, 256)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

class Bayesian_Linear(nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(Bayesian_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_logvar = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.param_init()
    
    def param_init(self):
        nn.init.uniform_(self.weight_mu, a=math.sqrt(6.0/(self.in_features + self.out_features)))
        self.weight_logvar.data.fill_(-6.0)
        
        if self.bias_mu is not None:
            self.bias_mu.data.fill_(0.0)
            self.bias_logvar.data.fill_(-6.0)
    
    def forward(self, x, mle=False):
        if mle:
            self.weight = self.weight_mu
            self.bias = self.bias_mu
        else:
            self.weight = sample_gaussian(self.weight_mu, self.weight_logvar)
            self.bias = sample_gaussian(self.bias_mu, self.bias_logvar)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Generator(nn.Module):
    def __init__(self, task_id=0):
        super(Generator, self).__init__()
        self.task_id = task_id
        self.latent_size = 256
        # Parameters
        self.headers = [Bayesian_Linear(self.latent_size, 500) for _ in range(task_id + 1)]
        self.fc = Bayesian_Linear(500, 28 * 28)
        self.layer_list = [self.fc]

    def set_task_id(self, task_id):
        self.task_id = task_id
    
    def generate(self, num = 1):
        """
        Sample a latent vector (z) 
        And return calculate obseravation (x)
        """
        z = torch.randn((num, self.latent_size)).squeeze()
        x = self.forward(z, mle=True)
        return z, x

    def forward(self, z, mle=False):
        """
        Returns Observation (x) given latent (z)
        """
        header = self.headers[self.task_id]
        h = F.relu(header(z, mle=mle))
        x = self.fc(h, mle=mle)
        return x.view(-1, 28, 28)

    def KL_term(self, old_model):
        kl_div = 0
        iterator = zip(self.layer_list, old_model.layer_list)

        for (layer, layer_prior) in iterator:
            # Detach old parameters from current graph (do not propagate grad)
            mu_prior_b = layer_prior.bias_mu.detach()
            mu_prior_w = layer_prior.weight_mu.detach()
            logvar_prior_b = layer_prior.bias_logvar.detach()
            logvar_prior_w = layer_prior.weight_logvar.detach()

            mu_b = layer.bias_mu
            mu_w = layer.weight_mu
            logvar_b = layer.bias_logvar
            logvar_w = layer.weight_logvar

            var_b = torch.exp(logvar_b)
            var_w = torch.exp(logvar_w)
            var_prior_b = torch.exp(logvar_prior_b)
            var_prior_w = torch.exp(logvar_prior_w)
            
            # Weights
            const_term = -0.5 * mu_w.numel()
            log_std_diff = 0.5 * (logvar_prior_w - logvar_w).sum()
            mu_diff_term = 0.5 * ((var_w + (mu_prior_w - mu_w)**2)/ var_prior_w).sum()
            kl_div += const_term + log_std_diff + mu_diff_term

            # Biases            
            const_term = -0.5 * mu_b.numel()
            log_std_diff = 0.5 * (logvar_prior_b - logvar_b).sum()
            mu_diff_term = 0.5 * ((var_b + (mu_prior_b - mu_b)**2) / var_prior_b).sum()
            kl_div += const_term + log_std_diff + mu_diff_term

        return kl_div

class VAE(nn.Module):
    def __init__(self, task_id = 0):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.generator = Generator(task_id)

    def set_task_id(self, task_id):
        self.generator.set_task_id(task_id)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = sample_gaussian(mu, logvar)
        x_recon = self.generator(z)
        return x_recon, mu, logvar

class VAE_loss(nn.Module):
    def __init__(self, vae, old_vae):
        super(VAE_loss, self).__init__()
        self.vae = vae
        self.old_vae = old_vae

    def forward(self, recon_x, x, mu, logvar):
        # Adjust the output range according to task id.
        bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        latent_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vcl_kl = self.vae.generator.KL_term(self.old_vae.generator)
        return bce + latent_kl + vcl_kl

"""
Test Cases
"""
def test0():
    generator = Generator(task_id=0)
    z = torch.randn(generator.latent_size)
    x = generator(z)
    
    # Plot Latent Distribution z
    import matplotlib.pyplot as plt
    plt.hist(z.detach().numpy())
    plt.show()

    # Plot image x
    from torchvision.transforms import ToPILImage
    transform = ToPILImage()
    im = transform(x)
    im.show()

def test1():
    vae = VAE(task_id=0)
    im = torch.randn(32, 1, 28, 28)
    recon_im, mu, logvar = vae.forward(im)

if __name__ == "__main__":
    test0()
    test1()