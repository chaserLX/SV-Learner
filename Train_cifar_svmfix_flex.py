from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_cifar import *
import dataloader_cifar as dataloader
from math import log2
from Contrastive_loss import *
from util_svmfix_cifar import svmfix_train, svmfix_train_flex
import collections.abc
from collections.abc import MutableMapping
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


## For plotting the logs
# import wandb
# wandb.init(project="noisy-label-project", entity="..")

## Arguments to pass 
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--lambda_tri', default=0.005, type=float)
parser.add_argument('--flex_threshold', default=0.5, type=float)
parser.add_argument('--gamma', default=2, type=float)
parser.add_argument('--epoch_start_svmfix', default=11, type=int)
parser.add_argument('--warm_up', default=10, type=int)
parser.add_argument('--margin', default=10, type=float)
parser.add_argument('--sample_threshold', default=0.8, type=float)


args = parser.parse_args()

## GPU Setup 
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
if args.dataset== 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path,train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path,train=False, download=True)

## Checkpoint Location
# folder = args.dataset + '_' + args.noise_mode + '_' + str(args.r) + 'test2_flex<0.9=0.9+avgconjs+Lu0+Rjs-g0.7+simclr' + '_DivideJSGMM_lamtri0.005_st0.7_ft0.7'  # + 'svmfix_LC+NCE'
folder = args.dataset + '_' + args.noise_mode + '_' + str(args.r) + 'SV-Learner + avg + u0 + without_SimCLR + svmfix_start_31'
model_save_loc = './checkpoint/' + '/new_ablation/' + folder

if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

## Log files
stats_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc +'/test_loss.txt','w')
train_acc = open(model_save_loc +'/train_acc.txt','w')
train_loss = open(model_save_loc +'/train_loss.txt','w')


## For Standard Training 
def warmup_standard(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs = net(inputs)               
        loss    = CEloss(outputs, labels)    

        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        else:   
            L = loss

        L.backward()  
        optimizer.step()                

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

## For Training Accuracy
def warmup_val(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            _, outputs  = net(inputs)               
            _, predicted = torch.max(outputs, 1)    
            loss    = CEloss(outputs, labels)    
            loss_x += loss.item()                      

            total   += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  
    
    train_loss.write(str(loss_x/(batch_idx+1)))
    train_acc.write(str(acc))
    train_acc.flush()
    train_loss.flush()

    return acc

## Test Accuracy
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs1 = net1(inputs)
            _, outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
            loss = CEloss(outputs, targets)  
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()  

    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    test_loss_log.write(str(loss_x/(batch_idx+1))+'\n')
    test_loss_log.flush()
    return acc


# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Calculate JSD
def Calculate_JSD(model1, model2, num_samples):
    model1.eval()
    model2.eval()
    JS_dist = Jensen_Shannon()
    JSD   = torch.zeros(num_samples)

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])

        ## Get the Prediction
        out = (out1 + out2)/2     

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist


    return JSD


def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if args.r >= 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

## Choose Warmup period based on Dataset
num_samples = 50000
if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=model_save_loc,log=stats_log, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True


## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 280, 2e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 280, 2e-4)

## Loss Functions
CEloss   = nn.CrossEntropyLoss()
CE       = nn.CrossEntropyLoss(reduction='none')
criterion_triplet = nn.MarginRankingLoss(margin=args.margin)
contrastive_criterion = SupConLoss()

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

## Resume from the warmup checkpoint 
model_name_1 = 'Net1_warmup.pth'
model_name_2 = 'Net2_warmup.pth'    

if args.resume:
    start_epoch = warm_up
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2))['net'])
else:
    start_epoch = 0


acc_hist = []
best_acc = 0
all_loss = [[],[]] # save the history of losses from two networks

classwise_acc_net1 = torch.zeros((args.num_class,)).cuda()
classwise_acc_net2 = torch.zeros((args.num_class,)).cuda()

class_flex_adjust1 = torch.ones((args.num_class,)).cuda()
class_flex_adjust2 = torch.ones((args.num_class,)).cuda()
# balanced_distri = torch.ones((args.num_class,)).cuda() * -1
gmm_cof1 = 1
gmm_cof2 = 1
jsd_cof1 = 1
jsd_cof2 = 1

## Warmup and SSL-Training 
for epoch in range(start_epoch,args.num_epochs+1):   
    test_loader = loader.run(0, 'test')
    eval_loader = loader.run(0, 'eval_train')   
    warmup_trainloader = loader.run(0,'warmup')
    
    ## Warmup Stage 
    if epoch<warm_up:       
        warmup_trainloader = loader.run(0, 'warmup')

        print('Warmup Model')
        warmup_standard(epoch, net1, optimizer1, warmup_trainloader)   

        print('\nWarmup Model')
        warmup_standard(epoch, net2, optimizer2, warmup_trainloader) 
    
    else:
        print("class_flex_adjust1", class_flex_adjust1)
        print("classwise_acc_net1", classwise_acc_net1)

        # print("kl_divergence1", kl_divergence(balanced_distri, class_flex_adjust1*-1))
        # Calculate JSD values and Filter Rate
        prob_js1 = Calculate_JSD(net2, net1, num_samples)
        threshold = torch.mean(prob_js1)

        # if threshold.item()>args.d_u:
        #     threshold = threshold - (threshold-torch.min(prob_js1))/args.tau
        # if epoch > args.epoch_start_svmfix:
        #     jsd_cof1 = (1 - threshold) / (1 - torch.min(prob_js1))

        SR_js = torch.sum(prob_js1 < threshold).item() / num_samples # * jsd_cof1

        prob_gmm1, all_loss[1] = eval_train(net2, all_loss[1])

        SR_gmm = np.sum(prob_gmm1 > args.sample_threshold) / num_samples # * gmm_cof1

        SR = (SR_js + SR_gmm) / 2.0
        # SR = (2 * SR_js + SR_gmm) / 3.0
        # SR = SR_js
        # print("gmm_cof1:", gmm_cof1)
        # print("jsd_cof1:", jsd_cof1)
        print("SR_gmm:", SR_gmm)
        print("SR_js:", SR_js)
        print("SR:", SR)

        print('Train Net1\n')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob_js1, class_flex_adjust=class_flex_adjust1, gmm_prob=prob_gmm1) # Uniform Selection
        # GMM
        # labeled_trainloader, unlabeled_trainloader = loader.run(SR_gmm, 'train', prob=prob_gmm1,
        #                                                         class_flex_adjust=class_flex_adjust1)
        classwise_acc_net1, class_flex_adjust1, gmm_cof1 = svmfix_train_flex(args, epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion_triplet,
                    classwise_acc_net1, contrastive_criterion)    # train net1


        # Calculate JSD values and Filter Rate
        prob_js2 = Calculate_JSD(net2, net1, num_samples)
        threshold = torch.mean(prob_js2)
        print("js_threshold", threshold)

        # if threshold.item()>args.d_u:
        #     threshold = threshold - (threshold-torch.min(prob_js2))/args.tau

        # if epoch > args.epoch_start_svmfix:
        #     jsd_cof2 = (1 - threshold) / (1 - torch.min(prob_js2))
        SR_js = torch.sum(prob_js2 < threshold).item() / num_samples  # * jsd_cof2

        prob_gmm2, all_loss[0] = eval_train(net1, all_loss[0])
        SR_gmm = np.sum(prob_gmm2 > args.sample_threshold) / num_samples # * gmm_cof2
        SR = (SR_js + SR_gmm) / 2.0
        # SR = (2 * SR_js + SR_gmm) / 3.0
        # SR = SR_js
        # print("kl_divergence1", kl_divergence(balanced_distri, class_flex_adjust2*-1))
        # print("gmm_cof2:", gmm_cof2)
        # print("jsd_cof2:", jsd_cof2)
        print("SR_gmm:", SR_gmm)
        print("SR_js:", SR_js)
        print("SR:", SR)
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob_js2, class_flex_adjust=class_flex_adjust2, gmm_prob=prob_gmm2)     # Uniform Selection
        # labeled_trainloader, unlabeled_trainloader = loader.run(SR_gmm, 'train', prob=prob_gmm2,
        #                                                         class_flex_adjust=class_flex_adjust2)
        classwise_acc_net2, class_flex_adjust2, gmm_cof2 = svmfix_train_flex(args, epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, criterion_triplet,
                     classwise_acc_net2, contrastive_criterion)      # train net1

    acc = test(epoch,net1,net2)
    # acc_hist.append(acc)
    scheduler1.step()
    scheduler2.step()

    if acc > best_acc:
        if epoch <warm_up:
            model_name_1 = 'Net1_warmup.pth'
            model_name_2 = 'Net2_warmup.pth'
        else:
            model_name_1 = 'Net1.pth'
            model_name_2 = 'Net2.pth'            

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': 'SGD',
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        best_acc = acc
        print("best_acc", best_acc)

