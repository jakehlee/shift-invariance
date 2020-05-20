import torch
import torch.nn as nn

# referencing "PyTorch: Custom nn Modules"
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout

class ConvShortout(nn.Module):

    def __init__(self, p=0.01):
        # p here is the short probability
        super(ConvShortout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("shortout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:
            if self.p == 0:
                return X
            # I'm assuming that X is (N, C, H, W).

            # 1. Maxpool the entire channel
            maxes = torch.nn.functional.max_pool2d(X, kernel_size=X.size()[2:])
            #maxes.detach()
            #maxes = nn.AdaptiveMaxPool2d((1,1))
            # maxes will be (N, C, 1, 1)
            maxes2 = torch.cat([maxes] * X.size(2), 2)
            maxes3 = torch.cat([maxes] * X.size(3), 3)
            # maxes will be back to (N, C, H, W).

            # 2. Dropout the entire thing
            # We're going to implement this from scratch here
            dist = torch.distributions.binomial.Binomial(probs=1-self.p)
            if X.is_cuda:
                mask = dist.sample(X.size()).cuda()
            else:
                mask = dist.sample(X.size())
            #mask = dist.sample(X.size())
            # mask will be (N, C, H, W)

            # 3. Replace dropped values with maxes
            # no rescaling for now...
            out = mask * X + (1-mask) * maxes3

            # let's try scaling - roughly p% are going to max, inceasing
            # magnitude. Decrease by multiplying the whole thing by keep prob
            X = out * (1 - self.p)
        return X
