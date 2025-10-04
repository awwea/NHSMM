import unittest
import torch
from torch.nn import ParameterDict, Parameter
from torch.distributions import Distribution
from nhsmm.hmm import MultinomialHMM #type:ignore


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.hmm = MultinomialHMM(2,3)

    def test_pdf_subclass(self):
        self.assertTrue(issubclass(self.hmm.pdf,Distribution))

    def test_dof_is_int(self):
        self.assertIsInstance(self.hmm.dof,int)

    def test_parameter_dict(self):
        self.assertIsInstance(self.hmm.sample_emission_params(),ParameterDict)

    def test_model_params(self):
        emission_params = self.hmm.sample_emission_params()
        for name,param in emission_params.items():
            self.assertIsInstance(param,Parameter)
            self.assertIsInstance(param.data,torch.Tensor)


