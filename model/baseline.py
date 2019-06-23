import torch
import torch.nn as nn
import torch.nn.functional as F

class Dnn(nn.Module):
    def __init__(self):
        super(Dnn, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MFVI_DNN(nn.Module):
    def __init__(self, MLE=False):
        super(MFVI_DNN, self).__init__()
        self.fc11 = nn.Linear(28*28, 256)
        self.fc12 = nn.Linear(28*28, 256)
        self.fc21 = nn.Linear(256, 256)
        self.fc22 = nn.Linear(256, 256)
        self.output1 = nn.Linear(256, 10)
        self.output2 = nn.Linear(256, 10)

        # List of submodules
        self.mu_list = [self.fc11, self.fc21, self.output1]
        self.logvar_list = [self.fc12, self.fc22, self.output2]

        self.MLE = MLE
        if self.MLE:
            self.const_init()

    def const_init(self):
        torch.nn.init.normal_(self.fc11.weight, std=0.1)
        torch.nn.init.normal_(self.fc11.bias, std=0.1)
        torch.nn.init.normal_(self.fc21.weight, std=0.1)
        torch.nn.init.normal_(self.fc21.bias, std=0.1)
        torch.nn.init.normal_(self.output1.weight, std=0.1)
        torch.nn.init.normal_(self.output1.bias, std=0.1)
        self.fc12.weight.data.fill_(0.0)
        self.fc12.bias.data.fill_(-6.0)
        self.fc22.weight.data.fill_(0.0)
        self.fc22.bias.data.fill_(-6.0)
        self.output2.weight.data.fill_(0.0)
        self.output2.bias.data.fill_(-6.0)

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + eps*std
        return sample
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        if not self.MLE:
            x = F.relu(self.reparametize(self.fc11(x), self.fc12(x)))
            x = F.relu(self.reparametize(self.fc21(x), self.fc22(x)))
            x = self.reparametize(self.output1(x), self.output2(x))
        else:
            x = F.relu(self.fc11(x))
            x = F.relu(self.fc21(x))
            x = self.output1(x)
        return F.log_softmax(x, dim = 1)

    def KL_term(self, old_model):
        kl_div = 0
        
        iterator = zip(self.mu_list, self.logvar_list, 
                old_model.mu_list, old_model.logvar_list)
        for (mu, logvar, mu_prior, logvar_prior) in iterator:
            if old_model.MLE:
                mu_prior_b = torch.zeros_like(mu_prior.bias, requires_grad=False)
                mu_prior_w = torch.zeros_like(mu_prior.weight, requires_grad=False)
                logvar_prior_b = torch.zeros_like(logvar_prior.bias, requires_grad=False)
                logvar_prior_w = torch.zeros_like(logvar_prior.weight, requires_grad=False)
            else:
                # Detach From current graph (do not propagate grad)
                mu_prior_b = mu_prior.bias.detach()
                mu_prior_w = mu_prior.weight.detach()
                logvar_prior_b = logvar_prior.bias.detach()
                logvar_prior_w = logvar_prior.weight.detach()

            var_b = torch.exp(logvar.bias)
            var_w = torch.exp(logvar.weight)
            var_prior_b = torch.exp(logvar_prior_b)
            var_prior_w = torch.exp(logvar_prior_w)
            
            # Weights
            const_term = -0.5 * mu.weight.numel()
            log_std_diff = 0.5 * (logvar_prior_w - logvar.weight).sum()
            mu_diff_term = 0.5 * ((var_w + (mu_prior_w - mu.weight)**2)/ var_prior_w).sum()
            kl_div += const_term + log_std_diff + mu_diff_term

            # Biases            
            const_term = -0.5 * mu.bias.numel()
            log_std_diff = 0.5 * (logvar_prior_b - logvar.bias).sum()
            mu_diff_term = 0.5 * ((var_b + (mu_prior_b - mu.bias)**2) / var_prior_b).sum()
            kl_div += const_term + log_std_diff + mu_diff_term

        return kl_div


class VCL_loss(nn.Module):
    def __init__(self, model, old_model):
        super(VCL_loss, self).__init__()
        self.model = model
        self.old_model = old_model

    def forward(self, output, target, reduction='mean'):
        normalizer = target.size(0) if reduction == 'mean' else 1
        nll = F.nll_loss(output, target, reduction=reduction)
        
        if self.old_model is None:
            return nll
        else:
            kl_term = self.model.KL_term(self.old_model) / normalizer
            return nll + kl_term


if __name__ == "__main__":
    model = MFVI_DNN(None)
