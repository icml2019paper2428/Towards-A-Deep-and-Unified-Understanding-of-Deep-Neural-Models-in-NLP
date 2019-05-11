"""
Toolkit that enables you to explain why your model make its dicision
"""

import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

class Interpreter(nn.Module):
    """
    Interpreter for interpret one instance.

    It will minimize the loss in Eqn.(7):

        $$L(\\sigma) = (||\\Phi(embed + \\epsilon) - \\Phi(embed)||_2^2) // (regularization) - rate * log(\\sigma)$$

    In our implementation, we use reparameterization trick to represent \epsilon ~ N(0, \sigma^2 I). i.e. \epsilon = scale * ratio * noise.
    Where noise ~ N(0, 1), scale is a hyper-parameter that controls the maximum value of \sigma^2, and ratio in (0, 1) is the learnable parameter.

    Params:
    ------
    x: ``torch.FloatTensor``, shape ``[length, dimension]``.
        The $$\\bf x$$ we studied. i.e. The input word embeddings.
    Phi: ``function``
        The $$\\Phi$$ we studied. A function whose input is x (the first parameter) and returns a hidden state (of type ``torch.FloatTensor``, of any shape)
    scale: ``float``
        The maximum size of \sigma^2. A hyper-parameter in reparameterization trick. The recommended value is 5 * Var[word_embedding_weight], 
        where word_embedding_weight is the word embedding weight in the model interpreted. Default: 0.5.
    rate: ``float``
        A hyper-parameter that balance the MLE Loss and Maximum Entropy Loss. Default: 0.1.
    regularization: ``Torch.FloatTensor``
        The regularization term, should be of the same shape as the output of \\Phi. If None is given, method will use the output to regularize itself. Default: None.
    words: `List[Str]`
        The input sentence, used for visualizing. If None is given, method will not show the words.
    """
    def __init__(self, x, Phi, scale=0.5, rate=0.1, regularization=None, words=None):
        super(Interpreter, self).__init__()
        self.s = x.size(0)
        self.d = x.size(1)
        self.ratio = nn.Parameter(torch.randn(self.s, 1), requires_grad=True)

        self.scale = scale
        self.rate = rate
        self.x = x
        self.Phi = Phi

        self.regular = regularization
        if self.regular is not None:
            self.regular = nn.Parameter(torch.tensor(self.regular), requires_grad=False)
        self.words = words
        if self.words is not None:
            assert self.s == len(words), 'the length of x should be of the same with the lengh of words'

    def forward(self):
        """
        Calculate loss:
        
            $$L = ||\Phi(embed + size * \sigma * \epsilon) - \Phi(embed)||_2^2 - rate * log(\sigma)$$

        Output:
        ------
        ``torch.FloatTensor``: a scalar, the target loss.
        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        x = self.x + 0.    # S * D
        x_tilde = x + ratios * torch.randn(self.s, self.d).to(x.device) * self.scale  # S * D
        s = self.Phi(x)  # D or S * D
        s_tilde = self.Phi(x_tilde)
        loss = (s_tilde - s) ** 2
        if self.regular is not None:
            loss = torch.mean(loss / self.regular ** 2)
        else:
            loss = torch.mean(loss) / torch.mean(s ** 2)

        return loss - torch.mean(torch.log(ratios)) * self.rate

    def optimize(self, iteration=5000, lr=0.01, show_progress=False):
        """
        Optimize the loss function

        Param:
        ------
        iteration: ``int``
            Total optimizing iteration
        lr: ``float``
            Learning rate
        show_progress: ``bool``
            Whether to show the learn progress
        """
        minLoss = None
        state_dict = None
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        func = (lambda x: x) if not show_progress else tqdm
        for e in func(range(iteration)):
            optimizer.zero_grad()
            loss = self()
            loss.backward()
            optimizer.step()
            if minLoss is None or minLoss > loss:
                state_dict = {k:self.state_dict()[k] + 0. for k in self.state_dict().keys()}
                minLoss = loss
        self.eval()
        self.load_state_dict(state_dict)

    def get_sigma(self):
        """
        Calculate and return the sigma

        Output:
        ------
        ``np.ndarray``: of shape ``[seqLen]``, the ``\sigma``.
        """
        ratios = torch.sigmoid(self.ratio)  # S * 1
        return ratios.detach().cpu().numpy()[:,0] * self.scale
    
    def visualize(self):
        """
        Visualize the information loss of every word.

        Output:
        ------
        None
        """
        sigma_ = self.get_sigma()
        fig, ax = plt.subplots()
        ax.imshow([sigma_], cmap='GnBu_r')
        ax.set_xticks(range(self.s))
        ax.set_xticklabels(self.words)
        ax.set_yticks([0])
        ax.set_yticklabels([''])
        plt.tight_layout()
        plt.show()