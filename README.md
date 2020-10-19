# Pytorch_qat_workflow

## Error

Getting the following error even after loading trained weights. 

```
AssertionError: min nan should be less than max nan
```

## Workflow

1. Created a custom [wrapper](https://github.com/SrivastavaKshitij/pytorch_qat_workflow/blob/ff4605c6f8a17959f62dca4d85286c136147328d/utilities.py#L52) for layers that need to be quantized with two modes: QAT on and off. 
2. Train the model with qat off for `x` number of epochs
3. Load the checkpoints correctly making sure that the keys do match using this [function](https://github.com/SrivastavaKshitij/pytorch_qat_workflow/blob/ff4605c6f8a17959f62dca4d85286c136147328d/utilities.py#L9)
4. Retrain the model with quantization on. 

This workflow gives per layer flexibility to quantize a layer or not. 

## Steps to reproduce the error:

1. Train the model with quantization off 

```
python train.py --m cnn
```

2. Load the checkpoint and starts retraining with quantization on 

```
python train.py --m cnn --netqat --partial_ckpt --load_ckpt /tmp/pytorch_exp/ckpt_{}.pth
```

#### Pointers

1. I tried lowering the learning rate to `1e-5` and even then i run into the error
2. I usually train for 30 epochs. If I load `26th checkpoint` or a ckpt from the later stage of the training when the model has almost converged, I get the error in the first epoch of retraining
3. If I load a checkpoint from early stages of training; `checkpoint <= 10`, then I can retrain for `4-5 epochs` but still run into the same error




