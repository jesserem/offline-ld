# Offline-LD
This repository contains the official implementation of the paper "Offline Reinforcement Learning for Learning to Dispatch for Job Shop Scheduling"

van Remmerden, J., Bukhsh, Z., & Zhang, Y. (2024). Offline Reinforcement Learning for Learning to Dispatch for Job Shop Scheduling. arXiv preprint arXiv:2409.10589.

If you use our code please cite our work as follows:
```
@article{van2024offline,
  title={Offline Reinforcement Learning for Learning to Dispatch for Job Shop Scheduling},
  author={van Remmerden, Jesse and Bukhsh, Zaharah and Zhang, Yingqian},
  journal={arXiv preprint arXiv:2409.10589},
  year={2024}
}
```
## Requirements
- Python 3.11
- Pytorch 2.1.2
- Gymnasium 0.28.1
- Minari 0.5.1

## Explanation of the code
### 1. 'create_dataset_from_data.py'
This file is used to create the datasets for training. In it four arguments are of importance, namely:
- **--dataset**: Is the path to the dataset npy file. These can be found in ./main_code/generated_data.
- **--folder**: Is the path to the CP solutions corresponding to the size of the instances. These can be found in ./results_cp_sat.
- **--noisy_prob**: Is the probability of a noisy action will be taken. If this is set above 0, then the dataset will be created with noisy actions, with each instance having a 50% probability of being noisy.
- **--dataset_name**: Is the name of the dataset that will be created.

### 2. Training
Training is done by either running "train_cql_qrdqn.py"" or "train_cql_sac.py". The arguments are explained in the code itself.

### 3. 'evaluate.py'
This file is used to evaluate the trained models, by either calling "eval_cql_qrdqn_all.py" or "eval_cql_sac_all.py". The arguments are explained in the code itself.

## Contact
If you have any questions or discover a bug, please contact me at via [email](mailto:j.v.remmerden@tue.nl).