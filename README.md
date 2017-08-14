# Anuvada: Interpretable Models for NLP using PyTorch

One of the common criticisms of deep learning has been it's black box nature. To address this issue, researchers have
developed many ways to visualise and explain the inference. Some examples would be attention in the case of RNN's,
activation maps, guided back propagation and occlusion (in the case of CNN's). This library is an ongoing effort to
provide a high-level access to such models relying on PyTorch.

## Installing

Clone this repo and add it to your python library path.

* [PyTorch](http://pytorch.org)
* [NumPy](http://numpy.org/)

### Getting started

```
# Import packages
import anuvada
import numpy as np
import torch
from anuvada.models.classification_attention_rnn import AttentionClassifier
# Create model object
acf = AttentionClassifier(vocab_size=50455,embed_size=300,gru_hidden=512,n_classes=62)
# Load data
x = torch.from_numpy(np.random.randint(0,500,(512,30)))
y = torch.from_numpy(np.random.randint(0,9,512))
# Call fit function
loss = acf.fit(x,y,epochs=5,batch_size=128)

```

## To do list

- [x] Implement Attention with RNN
- [ ] Implement Attention Visualisation
- [x] Implement working Fit Module
- [ ] Implement support for masking gradients in RNN
- [x] Implement a generic data set loader
- [ ] Implement CNN Classifier with feature map visualisation

## Acknowledgments

* https://github.com/henryre/pytorch-fitmodule