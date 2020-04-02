# ZB4199 Final Year Project: <br/> Deep Learning and The Diffusion Partial Differential Equation from simulated data of Drosophila embryos.

The code repository for Python programmes used for the Final Year Project (FYP).

## Items in this repository

1. Data_Generating_NoteBooks folder: Python notebooks to generate 2D datasets for DeepMod

   - **Diffusion_noisy.ipynb**: Notebook to produce noisy 2D data for Diffusion
   
   - **Diffusion_clean.ipynb**: Notebook to produce clean 2D data for Diffusion
   
   - **Diffusion_Decay_defaultBC.ipynb**: Notebook to produce clean/noisy 2D data for Diffusion-Decay with default boundary condition as stated in the FYP thesis (sections 5.2.1 to 5.2.2 of thesis).
   
   - **Diffusion_Decay_bicoidSurge.ipynb**: Notebook to produce clean/noisy 2D data for Diffusion-Decay with alternative boundary condition 1 as stated in the FYP thesis (section 5.2.3 of thesis).
   
   - **Diffusion_Decay_flux.ipynb**: Notebook to produce clean/noisy 2D data for Diffusion-Decay with alternative boundary condition 2 as stated in the FYP thesis (section 5.2.3 of thesis).

2. DeepMod_Training folder: Python programmes to train Deepmod algorithm

   - **deepmod_original.py**: Software pipeline to load data to train DeepMod.
   
   - **gpu_util.py**: Programme to search for GPU resources within MBI's computing infrastructure for Deep Learning.
   
   - **parameters.py**: Programme containing DeepMod's neural network architecture and Deep/Machine Learning hyper-parameters.
   
   - **pde_util.py**: Programme containing library of PDE terms for DeepMod and other relevant library parameters.
   
   - **result_util.py**: Programme to store results after training DeepMod.

3. Algorithm1.py: Python programme which contained an enhanced procedure of DeepMod's thresholding (section 5.3.1 of thesis).
4. Figure31.png: Image of Figure 31 for section 6.1 of the FYP thesis.
5. LR_expt folder: LR means <u>L</u>asso/<u>L</u>inear <u>R</u>egression. Python notebook which discusses DeepMod's normalisation issues (section 5.3.2 of thesis).

### Prerequisites

1. Python programming language version: 3.7.4.
2. Tensorflow version: 1.14.0.
3. Numpy version: 1.17.5.
4. Scikit-learn version: 0.22.1.
5. Pandas version: 1.0.0.
6. DeepMod (written in Tensorflow, Python): https://github.com/PhIMaL/DeePyMoD. 

## Author

**Ho Wei Chin (National University of Singapore)** 


## Acknowledgments

**Tricia Loo**, for the Forward-Time-Central-Space Python programme.

**Gert-Jan Both et al**, for their DeepMod algorithm.

