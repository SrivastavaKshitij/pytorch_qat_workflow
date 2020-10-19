import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import argparse
import os,sys 
import torch.optim as optim 
from cifar10 import Cifar10Loaders
from models import cnn
from utilities import calculate_accuracy, map_ckpt_names 
from parser import parse_args
import time
import torch.quantization as quantization
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

def main():
    args = parse_args()

    ## Create an output dir
    output_dir_path = args.od + args.en
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        dir_name=output_dir_path 
        tb_dirname = output_dir_path + '/tb_logdir'
    else:
        counter=1
        dir_name = output_dir_path
        new_dir_name = dir_name
        while os.path.exists(new_dir_name):
            new_dir_name = dir_name + "_" + str(counter)
            counter +=1 
        os.makedirs(new_dir_name)
        dir_name=new_dir_name
        tb_dirname = dir_name + "/tb_logdir" 

    print("===>> Output folder = {}".format(dir_name))
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
    
    loaders = Cifar10Loaders()
    train_loader = loaders.train_loader()
    test_loader = loaders.test_loader()

    if args.m == "vanilla_cnn":
        model = vanilla_cnn()
    elif args.m == "vanilla_cnn_qat":
        model = vanilla_cnn_qat()
    elif args.m == "cnn":
        if args.netqat:
            model=cnn(qat_mode=True)
            torch.quantization.prepare_qat(model,inplace=True)
        else:
            model=cnn()
    elif args.m == "shufflenet":
        model = models.shufflenet_v2_x1_0(pretrained=False)
    else:
        raise NotImplementedError("{} model not found".format(args.m))


    ## Instantiate tensorboard logs
    writer = SummaryWriter(tb_dirname)
    images, labels = iter(train_loader).next()
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('cifar-10', img_grid)
    #writer.add_graph(model,images)

    if args.cuda:
        model = model.cuda()
        if args.parallel:
            model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_test_accuracy=0
    print("===>> Training started")

    if args.load_ckpt:
        model.eval()
        checkpoint = torch.load(args.load_ckpt)
        if args.partial_ckpt:
            model_state = checkpoint['model_state_dict']
            new_state_dict = map_ckpt_names(checkpoint['model_state_dict'])
            model.load_state_dict(new_state_dict,strict=False)
        else:
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("===>>> Checkpoint loaded successfully from {} at epoch {} ".format(args.load_ckpt,epoch))
    if args.qm:
        #model = torch.quantization.quantize_dynamic(model, {torch.nn.Conv2d}, dtype=torch.qint8)
        model.eval()
        model.fuse_model()
        model.qconfig=torch.quantization.QConfig(activation=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,dtype=torch.quint8,reduce_range=False,qscheme=torch.per_tensor_symmetric,quant_min=0,quant_max=255),weight=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,dtype=torch.qint8,qscheme=torch.per_tensor_symmetric,quant_min=-127,quant_max=127))
        #model.qconfig=torch.quantization.get_default_qat_qconfig('fbgemm')
        print(model.qconfig)
        torch.quantization.prepare_qat(model, inplace=True)

    print(model)
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        running_loss=0.0
        start=time.time()
        model.train()
        for i, data in enumerate(train_loader,0):
            inputs, labels = data

            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss +=loss.item()
            if i %1000 == 999:
                writer.add_scalar('train_loss',running_loss/1000, epoch * len(train_loader) + i)
                writer.add_scalar('learning_rate',optimizer.param_groups[0]['lr'],epoch * len(train_loader) + i)
        
        if epoch > 0 and  epoch % args.lrdt == 0:
            print("===>> decaying learning rate at epoch {}".format(epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.94


        running_loss /= len(train_loader)
        end = time.time()
        test_accuracy = calculate_accuracy(model,test_loader)
        writer.add_scalar('test_accuracy',test_accuracy, epoch)

        ## Adding histograms after every epoch
        for tag, value in model.named_parameters():
            tag = tag.replace('.','/')
            writer.add_histogram(tag,value.data.cpu().numpy(),epoch)

        print("Epoch: {0} | Loss: {1} | Test accuracy: {2}| Time Taken (sec): {3} ".format(epoch+1, np.around(running_loss,6), test_accuracy, np.around((end-start),4)))

        best_ckpt_filename = dir_name + "/ckpt_" + str(epoch) +'.pth'
        ##Save the best checkpoint
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                }, best_ckpt_filename)
    writer.close()
    print("Training finished")

if __name__ == "__main__":
    main()
