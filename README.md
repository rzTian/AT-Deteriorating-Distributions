# Adversarial Training May Induce Deteriorating Distributions


## Overview
This repository contains the code implementation for the paper [Adversarial Training May Induce Deteriorating Distributions](https://openreview.net/forum?id=8ixaK3sMhv) .
Key takeaways:
- We discover that adversarial training (AT) may induce deteriorating distributions.
- We empirically observed a correlation between the deteriorating behavior of the induced distributions and robust overfitting.
- Our findings highlight the critical role played by the perturbation operator in AT.

## Reproducing the Induced Distribution Experiments (IDEs)

To reproduce the experiments on the CIFAR-10 dataset, please follow these steps::
- Step 1: Perform adversarial training. Run
  ```
   python adv_train_cifar.py    --train_mode  'adv_train'  --dataset 'cifar10'
  
  ```
   This will create a folder named ```cifar10_chkpts``` (if it doesn't exist) where model checkpoints will be saved.
  
- Step 2: Using the saved checkpoints, generate perturbed versions of the train and test sets:
  ```
  python Generate_dataset.py
  
  ```
  The perturbed datasets will be stored in the folder ```cifar10_perturbed```

- Step 3: Train new models from scratch on the perturbed datasets by specifying the checkpoint ID:

  ```
  python IDE_cifar.py    --load_chkpt [Model Checkpoint id]  --train_mode  'std_train'
  ```
  The training results for these IDEs are saved in the folder [IDE_results_cifar10](https://github.com/rzTian/AT-Deteriorating-Distributions/tree/main/IDE_results_cifar10)
   
