# Evaluation

Run the commands below to evaluate a pretrained model.

```shell script
python eval.py --cfg configs/config_lstm_train.yaml
```

Change the `TRAIN.PRETRAINED` field of the config file to the checkpoint you would like to evaluate.
You should be able to obtain the output below:

```shell script
# TRAIN.PRETRAINED = 'data/vibe_data/vibe_model_wo_3dpw.pth.tar'
...Evaluating on 3DPW test set...
MPJPE: 93.5881, PA-MPJPE: 56.5608, PVE: 113.4118, ACCEL: 27.1242, ACCEL_ERR: 27.9877

...ResNet50...
MPJPE: 524.0020, PA-MPJPE: 224.8403, PVE: 559.2442, ACCEL: 696.0471, ACCEL_ERR: 778.6877,

...ResNeXt50...
MPJPE: 524.0105, PA-MPJPE: 234.9845, PVE: 556.1211, ACCEL: 682.9168, ACCEL_ERR: 753.7048,

...swin_b...
MPJPE: 486.6182, PA-MPJPE: 238.9666, PVE: 527.4085, ACCEL: 621.9053, ACCEL_ERR: 729.1726,

...spin...
MPJPE: 96.0632, PA-MPJPE: 64.2116, PVE: 112.2971, ACCEL: 252.3495, ACCEL_ERR: 141.7411,

...Geo...
MPJPE: 82.0473, PA-MPJPE: 55.5407, ACCEL: 27.5803, PVE: 98.8396, ACCEL_ERR: 28.9801,

```
