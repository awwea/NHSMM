# Chad Hidden Markov Models (ChadHMM)

> **NOTE:**
> This package is still in its early stages, documentation might not reflect every method mentioned above, please feel free to contribute and make this more coherent.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#references">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This repository was created as an attempt to learn and recreate the parameter estimation for Hidden Markov Models using PyTorch library. Included are models with Categorical and Gaussian emissions, for both Hidden Markov Models (HMM) and Hidden Semi-Markov Models(HSMM). As en extension I am trying to include models where the parameter estimation depends on certain set of external variables, these models are referred to as Contextual HMM or Parametric/Conditional HMM where the emission probabilities/distribution paramters are influenced by the context.

The documentation on the parameter estimation and model description is captured in [docs](https://github.com/GarroshIcecream/ChadHMM//tree/master/docs) folder. Furthermore, there are [examples](https://github.com/GarroshIcecream/ChadHMM//tree/master/tests) of the usage, especially on the financial time series, focusing on the sequence prediction but also on the possible interpretation of the model parameters.

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

1. Clone the repo
   ```sh
   git clone https://github.com/GarroshIcecream/ChadHMM.git
   ```
2. Install from PyPi
   ```sh
   pip install chadhmm
   ```

<!-- USAGE EXAMPLES -->
## Usage

Please refer to the [docs](https://github.com/GarroshIcecream/ChadHMM//tree/master/docs) for more detailed guide on how to create, train and predict sequences using Hidden Markov Models. There is also a section dedicated to visualizing the model parameters as well as its sequence predictions.

<!-- ROADMAP -->
## Roadmap

- [ ] Hidden Semi Markov Model numerical instability
    - [ ] Fix computation of posteriors 
    - [x] Fix mean and covariance update in HSMM
- [X] K-Means for Gaussian means initialization
- [ ] Improve the documentation with examples
    - [ ] Application on financial time series prediction
- [ ] Code base refactor, abstractions might be confusing
- [ ] Integration of contextual model for continous distributions
  - [ ] Time dependent context to be implemented
  - [ ] Contextual Variables for covariances using GEM (Genereliazed Expectation Maximization algo)
- [ ] Contextual variables for Categorical emissions
- [ ] Support for wider range of emissions distributions
- [ ] More visual tools for model interpretations
- [ ] Performance improvements
    - [ ] JIT or else
    - [ ] Improved Tensor ops
    - [ ] CUDA optimizations

See the [open issues](https://github.com/GarroshIcecream/ChadHMM/issues) for a full list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue.
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a [Pull Request](https://github.com/GarroshIcecream/ChadHMM/pulls)

<!-- REFERENCES -->
## References

Implementations are based on:

- Hidden Markov Models (HMM):
   - ["A tutorial on hidden Markov models and selected applications in speech recognition"](https://ieeexplore.ieee.org/document/18626) by Lawrence Rabiner from Rutgers University

- Hidden Semi-Markov Models (HSMM):
   - ["An efficient forward-backward algorithm for an explicit-duration hidden Markov model"](https://www.researchgate.net/publication/3342828_An_efficient_forward-backward_algorithm_for_an_explicit-duration_hidden_Markov_model) by Hisashi Kobayashi from Princeton University

- Contextual HMM and HSMM:
  - ["Contextual Hidden Markov Models"](https://www.researchgate.net/publication/261490802_Contextual_Hidden_Markov_Models) by Thierry Artieres from Ecole Centrale de Marseille

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
















