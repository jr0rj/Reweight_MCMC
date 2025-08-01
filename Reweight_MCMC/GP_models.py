import gpytorch
import torch





class BaseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(BaseGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood

        
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def optimize(self, training_iter = 100,output = False):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if output ==True:
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
                loss.backward()
                print('Iter %d/%d - Loss: %f   lengthscale: %f  outputscale: %f  noise: %f' % (
                    i + 1, training_iter, loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.covar_module.outputscale.item(),
                    self.likelihood.noise.item(),
                ))
                optimizer.step()
            
        else:
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()


class 1DGPModel(BaseGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(1DGPModel, self).__init__(train_x, train_y, likelihood, kernel)
    

    def plot(self,observed = True,limit = 'none',**kwargs):#xlab = 'x',ylab = 'y', title = ''):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            xrange = torch.linspace(min(self.train_x), max(self.train_x), 1000,dtype=torch.float32)
            observed_pred = self(xrange)
        mean = observed_pred.mean.numpy()
        lower, upper = observed_pred.confidence_region()
        lower = lower.numpy()
        upper = upper.numpy()
        if limit == 'logsumexp':
            mean,upper,lower = GP_logsumexp(mean,upper,lower,10000)
        if limit =='hardmax':
            mean,upper,lower = GP_hardmax(mean,upper,lower)
        with torch.no_grad():#torch.nograd skips gradianet calcs to save time, don'tneed grads in forward calcs
            f, ax = plt.subplots(1, 1)
            ax.grid()
            if observed == True:
                ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'x',color='green')
            ax.plot(xrange, mean, 'b')
            ax.fill_between(xrange, lower, upper, alpha=0.4,color='orange')
            ax.legend(['Observed Data','Mean', 'Confidence'])
            ax.set_xlabel(kwargs.get("ylabel", "y"))
            ax.set_ylabel(kwargs.get("xlabel", "x"))
            ax.set_title(kwargs.get("title",""))
            #plt.show()

    
    def evaluate(self,x,limit = 'none'):
        self.eval()
        self.likelihood.eval()
        x = torch.tensor(x,dtype = torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
        mean = observed_pred.mean.numpy()
        std = np.sqrt(observed_pred.variance.numpy())
        if limit == 'logsumexp':
            return softmax(mean,10000),std
        elif limit == 'hardmax':
            return hardmax(mean),std
        return mean, std
    
    
    def eval_mean(self,x, limit = 'none'):
        self.eval()
        self.likelihood.eval()
        x = torch.tensor([[x]],dtype = torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
        mean = observed_pred.mean.numpy()
        if limit == 'logsumexp':
            return softmax(mean,10000)
        if limit == 'hardmax':
            return hardmax(mean)
        return mean
    
    def eval_mean_vec(self,x, limit = 'none'):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self(x)
        mean = observed_pred.mean
        if limit == 'logsumexp':
            return softmax2(mean,10000)
        if limit == 'hardmax':
            return hardmax(mean)
        return mean
    
    def optimize(self, training_iter = 100,output = False):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if output ==True:
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
                loss.backward()
                print('Iter %d/%d - Loss: %f   lengthscale: %f  outputscale: %f  noise: %f' % (
                    i + 1, training_iter, loss.item(),
                    self.covar_module.base_kernel.lengthscale.item(),
                    self.covar_module.outputscale.item(),
                    self.likelihood.noise.item(),
                ))
                optimizer.step()
            
        else:
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()



