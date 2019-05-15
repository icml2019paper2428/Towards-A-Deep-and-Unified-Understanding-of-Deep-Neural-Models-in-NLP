# Towards a Deep and Unified Understanding of Deep Neural Models in NLP

Code implementation of paper *Towards a Deep and Unified Understanding of Deep Neural Models in NLP*

## Environment

In order to use our code, you need to install following softwares/packages:

```
python>=3.5
pytorch-pretrained-bert==0.2.0
torch==0.4.1
matplotlib
numpy
tqdm
```

You may also need to install jupyter notebook to visit our [torturial](Torturial.ipynb).

## How to use

The important class we need to utilize is the `Interpreter` in [Interpreter.py](Interpreter.py). Given any input word embeddings and a forward function $\Phi$ that transforms the word embeddings to a hidden state, Interpreter helps understand how much each input word contributes to the hidden state. Suppose the $\Phi$, the input $\bf x$ and the input words are defined as:
```
import torch

x = torch.randn(5,256) / 10
words = ['1','2','3','4','5']

def Phi(x):
    W = torch.tensor([10., 20., 5., -20., -10.]).to(device)
    return W @ x
```

To explain this case, we need to initialize an `Interpreter` class, and pass $\bf x$ and $\Phi$ to it:
```
from Interpreter import Interpreter

interpreter = Interpreter(x=x, Phi=Phi, words=words)
```
Then, we need the interpreter to optimize itself by minimizing the loss function in paper.
```
interpreter.optimize(iteration=5000, lr=0.01, show_progress=True)
```
After optimizing, we can get the best sigma:
```
interpreter.get_sigma()
```
the result will be something like:
```
array([0.2203494 , 0.19501153, 0.19684102, 0.28645414, 0.24175803,
    0.25448853, 0.23727599, 0.18001308, 0.30041832, 0.28238717,
    0.29902193, 0.16674334, 0.32668313, 0.4206538 ], dtype=float32)
```
Every sigma stands for the change limit of input without changing hidden state too much. The smaller the sigma is, the more this input word contributes to the hidden state.

Now, we can get the explanation by calling the visualize function:
```
interpreter.visualize()
```
Then, we can get results below:

![](img/result.PNG)

which means that the second and forth words are most important to $\Phi$, which is reasonable because the weight of them are larger.

## Explain a certain layer in any saved pytorch model

We provide an example on how to use our method to explain a saved pytorch model(*pre-trained BERT model in our case*) [here](Torturial_BERT.ipynb). 
> NOTE: This result may not be consistent with the result in the paper because  we use the pre-trained BERT model directly for simplicity, while the BERT model we use in paper is fine-tuned on specific dataset like SST-2.