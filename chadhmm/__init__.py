"""
ChadHMM
======

Ultra Chad Implementation of Hidden Markov Models in Pytorch (available only to true sigma males)

But seriously this package needs you to help me make it better. I'm not a professional programmer, I'm just a guy who likes to code. 
If you have any suggestions, please let me know. I'm open to all ideas.
"""

# Import HMM objects
from .hmm import CategoricalHMM, GaussianHMM, GaussianMixtureHMM
from .hsmm import CategoricalHSMM, GaussianHSMM, GaussianMixtureHSMM
from .mixture_models import GaussianMixtureModel
from .stochastic_matrix import StochasticTensor