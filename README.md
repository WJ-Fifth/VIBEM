# The Exploration and Improvement of VIBE model


## Implementation
- Collect and load model datasets。
- Evaluate the performance of the original paper model on existing datasets.
- Use a different Backbone to replace the original Reset50 and test it under the same conditions (pre-trained in ImageNet 1K).
- Modify the GRU of VIBE's Temporal Encoder. It is replaced by a more refined LSTM, and the corresponding experimental tests are carried out.
- Modify the loss function of Motion Discriminator.
- The training module code has been optimized, and the original Trainer and evaluation codes are seriously redundant and merged and optimized.

## Source Code
- lib.core
  - config.py: The config file is only used to set configuration information, fine-tune the code, and add the ability to judge backbones, additional datasets, lstm, and new loss functions.
  - loss.py: Implemented and added the geometric GAN loss and the L2 Sensitive GAN loss proposed in this paper.
  - main.py: The training and evaluation module code is integrated and refactored, reducing code redundancy and improving code usability.
- lib.utils is the toolkit required for model implementation, provided for VIBE authors.
- lib.data_utils part of the code is provided by the author of VIBE. This project improves this module and adds new datasets to the original data processing module: HumanML3D and SSP-3D.
- lib.dataset part of the code is provided by the author of VIBE. We mainly implement the loading and use of SSP-3D and HumanML3D datasets. And solve the problem that the 3DPW dataset cannot be used for training.
- lib.models 
  - backbone.py: Consolidate feature extractors into one code file for use. In addition to the ResNet50 model in the spin used in the original text, the swin transformer and the ResNeXt model are also used.
  - swin.py It mainly includes the implementation of the swin transformer, and finally adopts the implementation method in the timm library due to the convenience of using the pre-training model.
  - vibe_lstm.py: Implement LSTM-based VIBEM model。
  - motion_discriminator_new.py: Implement an action discriminator module for VIBEM (LSTM).
  - attention.py, smple.py, vibe.py and motion_discriminator.py is the original model code

## Getting Started
VIBE has been implemented and tested on Ubuntu 18.04 with python >= 3.7. It supports both GPU and CPU inference.
If you don't have a suitable device, try running our Colab demo. 


Install the requirements using `virtualenv` or `conda`:
```bash
# pip
source scripts/install_pip.sh

# conda
source scripts/install_conda.sh
```

## Training
Run the commands below to start training:

```shell script
# prepare the training, evaluation dataset
source scripts/prepare_training_data.sh
# prepare the evaluation and pre-trained backbone in VIBE.
source scripts/prepare_data.sh

# THE config file save in configs
python train.py --cfg configs/config.yaml
```
## Evaluation

Here we compare VIBE and VIBEM result. Evaluation metric is
Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) in mm.

Please download our checkpoint about VIBEM on Google Drive [here](https://drive.google.com/file/d/1J77gZxEQ_Ge5PROuzDVHE2pN-rpdoReS/view?usp=share_link), 
and save the checkpoint in data/vibe_data/

| Models      | 3DPW &#8595; | MPI-INF-3DHP &#8595; | SSP_3D &#8595; |
| ----------- | :----------: | :------------------: | :------------: |
| VIBE        |     63.8     |         67.2         |      64.2      |
| VIBE_l2-SL  |     58.6     |         68.5         |      63.1      |
| VIBEM_l2-SL |   **55.2**   |       **65.9**       |    **62.6**    |


```shell script

# THE config file save in configs
# if you want to use ssp_3d dataset, please use the len_seq = 2 because the length of ssp_3d is really small.
python eval.py --cfg configs/config_lstm_eval.yaml
```
