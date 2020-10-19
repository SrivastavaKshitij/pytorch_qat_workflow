# Pytorch_qat_workflow

## Error

Getting the following error even after loading trained weights. 

```
AssertionError: min nan should be less than max nan
```

## Environment      
  
`Pytorch NGC container 20.09`
   
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
4. According to this [github issue](https://github.com/pytorch/pytorch/issues/41791), we are more likely to see the issue if we are training from scratch. So I made sure that the weights of Conv and BN stats were being loaded correctly. That brings me to other issue

As a side experiment, I changed `strict=True` at this line in [train.py](https://github.com/SrivastavaKshitij/pytorch_qat_workflow/blob/9a0509a49a878e6b591e80fb1871ca8a14491091/train.py#L89)

and I added the following print statement inside my ngc container after `line 963` in the file `vim /opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py` 

```
963             if key in state_dict:
964                 print("Key present in the state_dict",key)
```

and even though it prints `Key present in the state_dict layer1.qconv.bn.weight` , the code still complains that `Missing key(s) in state_dict: "layer1.qconv.bn.weight",...,...` . 


I would expect it to complain about other keys not present in the ckpt such as `layer1.qconv.bn.activation_post_process.zero_point` as QAT was off , however, I was not expecting it to complain about keys present in the state dict. 

So I dont know why this is happening as well. Below is the complete log:

```
Key present in the state_dict layer1.qconv.weight                                                                                                                                                             
Key present in the state_dict layer1.qconv.bn.weight                                                                                                                                                          
Key present in the state_dict layer1.qconv.bn.bias                                                                                                                                                            
Key present in the state_dict layer1.qconv.bn.running_mean                                                                                                                                                    
Key present in the state_dict layer1.qconv.bn.running_var                                                                                                                                                     
Key present in the state_dict layer1.qconv.bn.num_batches_tracked                                                                                                                                             
Key present in the state_dict layer2.qconv.weight                                                                                                                                                             
Key present in the state_dict layer2.qconv.bn.weight                                                                                                                                                          
Key present in the state_dict layer2.qconv.bn.bias                                                                                                                                                            
Key present in the state_dict layer2.qconv.bn.running_mean                                                                                                                                                    
Key present in the state_dict layer2.qconv.bn.running_var                                                                                                                                                     
Key present in the state_dict layer2.qconv.bn.num_batches_tracked                                                                                                                                             
Key present in the state_dict layer3.qconv.weight                                                                                                                                                             
Key present in the state_dict layer3.qconv.bn.weight                                                                                                                                                          
Key present in the state_dict layer3.qconv.bn.bias                                                                                                                                                            
Key present in the state_dict layer3.qconv.bn.running_mean                                                                                                                                                    
Key present in the state_dict layer3.qconv.bn.running_var                                                                                                                                                     
Key present in the state_dict layer3.qconv.bn.num_batches_tracked                                                                                                                                   
Key present in the state_dict layer4.qconv.weight
Key present in the state_dict layer4.qconv.bn.weight
Key present in the state_dict layer4.qconv.bn.bias
Key present in the state_dict layer4.qconv.bn.running_mean
Key present in the state_dict layer4.qconv.bn.running_var
Key present in the state_dict layer4.qconv.bn.num_batches_tracked
Key present in the state_dict fcs.0.weight
Key present in the state_dict fcs.0.bias
Key present in the state_dict fcs.2.weight
Key present in the state_dict fcs.2.bias
Key present in the state_dict fcs.4.weight
Key present in the state_dict fcs.4.bias

RuntimeError: Error(s) in loading state_dict for cnn:
        Missing key(s) in state_dict: "layer1.qconv.bn.weight", "layer1.qconv.bn.bias", "layer1.qconv.bn.running_mean", "layer1.qconv.bn.running_var", "layer1.qconv.bn.num_batches_tracked", "layer1.qconv.bn
.activation_post_process.scale", "layer1.qconv.bn.activation_post_process.zero_point", "layer1.qconv.bn.activation_post_process.fake_quant_enabled", "layer1.qconv.bn.activation_post_process.observer_enabled
", "layer1.qconv.bn.activation_post_process.scale", "layer1.qconv.bn.activation_post_process.zero_point", "layer1.qconv.bn.activation_post_process.activation_post_process.min_val", "layer1.qconv.bn.activati
on_post_process.activation_post_process.max_val", "layer1.qconv.bn.activation_post_process.activation_post_process.min_val", "layer1.qconv.bn.activation_post_process.activation_post_process.max_val", "layer
1.qconv.activation_post_process.scale", "layer1.qconv.activation_post_process.zero_point", "layer1.qconv.activation_post_process.fake_quant_enabled", "layer1.qconv.activation_post_process.observer_enabled",
 "layer1.qconv.activation_post_process.scale", "layer1.qconv.activation_post_process.zero_point", "layer1.qconv.activation_post_process.activation_post_process.min_val", "layer1.qconv.activation_post_proces
s.activation_post_process.max_val", "layer1.qconv.activation_post_process.activation_post_process.min_val", "layer1.qconv.activation_post_process.activation_post_process.max_val", "layer1.qconv.weight_fake_
quant.scale", "layer1.qconv.weight_fake_quant.zero_point", "layer1.qconv.weight_fake_quant.fake_quant_enabled", "layer1.qconv.weight_fake_quant.observer_enabled", "layer1.qconv.weight_fake_quant.scale", "la
yer1.qconv.weight_fake_quant.zero_point", "layer1.qconv.weight_fake_quant.activation_post_process.min_val", "layer1.qconv.weight_fake_quant.activation_post_process.max_val", "layer1.qconv.weight_fake_quant.
activation_post_process.min_val", "layer1.qconv.weight_fake_quant.activation_post_process.max_val", "layer2.qconv.bn.weight", "layer2.qconv.bn.bias", "layer2.qconv.bn.running_mean", "layer2.qconv.bn.running
_var", "layer2.qconv.bn.num_batches_tracked", "layer2.qconv.bn.activation_post_process.scale", "layer2.qconv.bn.activation_post_process.zero_point", "layer2.qconv.bn.activation_post_process.fake_quant_enabl
ed", "layer2.qconv.bn.activation_post_process.observer_enabled", "layer2.qconv.bn.activation_post_process.scale", "layer2.qconv.bn.activation_post_process.zero_point", "layer2.qconv.bn.activation_post_proce
ss.activation_post_process.min_val", "layer2.qconv.bn.activation_post_process.activation_post_process.max_val", "layer2.qconv.bn.activation_post_process.activation_post_process.min_val", "layer2.qconv.bn.ac
tivation_post_process.activation_post_process.max_val", "layer2.qconv.activation_post_process.scale", "layer2.qconv.activation_post_process.zero_point", "layer2.qconv.activation_post_process.fake_quant_enab
led", "layer2.qconv.activation_post_process.observer_enabled", "layer2.qconv.activation_post_process.scale", "layer2.qconv.activation_post_process.zero_point", "layer2.qconv.activation_post_process.activati
on_post_process.min_val", "layer2.qconv.activation_post_process.activation_post_process.max_val", "layer2.qconv.activation_post_process.activation_post_process.min_val", "layer2.qconv.activation_post_proces
s.activation_post_process.max_val", "layer2.qconv.weight_fake_quant.scale", "layer2.qconv.weight_fake_quant.zero_point", "layer2.qconv.weight_fake_quant.fake_quant_enabled", "layer2.qconv.weight_fake_quant.
observer_enabled", "layer2.qconv.weight_fake_quant.scale", "layer2.qconv.weight_fake_quant.zero_point", "layer2.qconv.weight_fake_quant.activation_post_process.min_val", "layer2.qconv.weight_fake_quant.acti
vation_post_process.max_val", "layer2.qconv.weight_fake_quant.activation_post_process.min_val", "layer2.qconv.weight_fake_quant.activation_post_process.max_val", "layer3.qconv.bn.weight", "layer3.qconv.bn.b
ias", "layer3.qconv.bn.running_mean", "layer3.qconv.bn.running_var", "layer3.qconv.bn.num_batches_tracked", "layer3.qconv.bn.activation_post_process.scale", "layer3.qconv.bn.activation_post_process.zero_poi
nt", "layer3.qconv.bn.activation_post_process.fake_quant_enabled", "layer3.qconv.bn.activation_post_process.observer_enabled", "layer3.qconv.bn.activation_post_process.scale", "layer3.qconv.bn.activation_po
st_process.zero_point", "layer3.qconv.bn.activation_post_process.activation_post_process.min_val", "layer3.qconv.bn.activation_post_process.activation_post_process.max_val", "layer3.qconv.bn.activation_post
_process.activation_post_process.min_val", "layer3.qconv.bn.activation_post_process.activation_post_process.max_val", "layer3.qconv.activation_post_process.scale", "layer3.qconv.activation_post_process.zero
_point", "layer3.qconv.activation_post_process.fake_quant_enabled", "layer3.qconv.activation_post_process.observer_enabled", "layer3.qconv.activation_post_process.scale", "layer3.qconv.activation_post_proce
ss.zero_point", "layer3.qconv.activation_post_process.activation_post_process.min_val", "layer3.qconv.activation_post_process.activation_post_process.max_val", "layer3.qconv.activation_post_process.activati
on_post_process.min_val", "layer3.qconv.activation_post_process.activation_post_process.max_val", "layer3.qconv.weight_fake_quant.scale", "layer3.qconv.weight_fake_quant.zero_point", "layer3.qconv.weight_fa
ke_quant.fake_quant_enabled", "layer3.qconv.weight_fake_quant.observer_enabled", "layer3.qconv.weight_fake_quant.scale", "layer3.qconv.weight_fake_quant.zero_point", "layer3.qconv.weight_fake_quant.activati
on_post_process.min_val", "layer3.qconv.weight_fake_quant.activation_post_process.max_val", "layer3.qconv.weight_fake_quant.activation_post_process.min_val", "layer3.qconv.weight_fake_quant.activation_post_
process.max_val", "layer4.qconv.bn.weight", "layer4.qconv.bn.bias", "layer4.qconv.bn.running_mean", "layer4.qconv.bn.running_var", "layer4.qconv.bn.num_batches_tracked", "layer4.qconv.bn.activation_post_pro
cess.scale", "layer4.qconv.bn.activation_post_process.zero_point", "layer4.qconv.bn.activation_post_process.fake_quant_enabled", "layer4.qconv.bn.activation_post_process.observer_enabled", "layer4.qconv.bn.
activation_post_process.scale", "layer4.qconv.bn.activation_post_process.zero_point", "layer4.qconv.bn.activation_post_process.activation_post_process.min_val", "layer4.qconv.bn.activation_post_process.acti
vation_post_process.max_val", "layer4.qconv.bn.activation_post_process.activation_post_process.min_val", "layer4.qconv.bn.activation_post_process.activation_post_process.max_val", "layer4.qconv.activation_p
ost_process.scale", "layer4.qconv.activation_post_process.zero_point", "layer4.qconv.activation_post_process.fake_quant_enabled", "layer4.qconv.activation_post_process.observer_enabled", "layer4.qconv.activ
ation_post_process.scale", "layer4.qconv.activation_post_process.zero_point", "layer4.qconv.activation_post_process.activation_post_process.min_val", "layer4.qconv.activation_post_process.activation_post_pr
ocess.max_val", "layer4.qconv.activation_post_process.activation_post_process.min_val", "layer4.qconv.activation_post_process.activation_post_process.max_val", "layer4.qconv.weight_fake_quant.scale", "layer
4.qconv.weight_fake_quant.zero_point", "layer4.qconv.weight_fake_quant.fake_quant_enabled", "layer4.qconv.weight_fake_quant.observer_enabled", "layer4.qconv.weight_fake_quant.scale", "layer4.qconv.weight_fa
ke_quant.zero_point", "layer4.qconv.weight_fake_quant.activation_post_process.min_val", "layer4.qconv.weight_fake_quant.activation_post_process.max_val", "layer4.qconv.weight_fake_quant.activation_post_proc
ess.min_val", "layer4.qconv.weight_fake_quant.activation_post_process.max_val".

```

