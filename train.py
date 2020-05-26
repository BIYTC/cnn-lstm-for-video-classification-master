import shutil
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.rmsprop import RMSprop
from tqdm import tqdm
from utils import AverageTracker
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import time
from utils import LoadImages



class Train:
    def __init__(self, model, trainloader, valloader, args):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.args = args
        self.start_epoch = 0
        self.best_top1 = 0.0

        self.test_best_top1 = 0.0
        self.batch_size = args.batch_size
        # self.mean = args.mean
        # self.std = args.std

        # Loss function and Optimizer
        self.loss = None
        self.optimizer = None
        self.create_optimization()

        # Model Loading
        self.load_pretrained_model()
        self.load_checkpoint(self.args.resume_from)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=args.summary_dir)
        # if not self.args.visual_mode:
        # save the graph for the neural network model
        # graph_input = torch.rand(args.batch_size, args.num_channels, args.img_height, args.img_width)
        # if self.args.cuda:
        #     graph_input = graph_input.cuda(self.args.cuda_select)
        # self.summary_writer.add_graph(model, (graph_input,))

    def train(self):
        for cur_epoch in range(self.start_epoch, self.args.num_epochs):

            # Initialize tqdm
            tqdm_batch = tqdm(self.trainloader,
                              desc="Epoch-" + str(cur_epoch) + "-")

            # Learning rate adjustment
            self.adjust_learning_rate(self.optimizer, cur_epoch)

            # Meters for tracking the average values
            loss, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker()

            # Set the model to be in training mode (for dropout and batchnorm)
            self.model.train()

            for iters, (data, target) in enumerate(tqdm_batch):

                if self.args.cuda:
                    data, target = data.cuda(device=self.args.cuda_select, non_blocking=self.args.async_loading), \
                                   target.cuda(device=self.args.cuda_select, non_blocking=self.args.async_loading)
                    # data, target = data.cuda(), target.cuda()
                data_var, target_var = Variable(data), Variable(target)

                # Forward pass
                output = self.model(data_var)
                cur_loss = self.loss(output, target_var)

                # Optimization step
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step()

                # Top-1 and Top-5 Accuracy Calculation
                cur_acc1, cur_acc5 = self.compute_accuracy(output.data, target, topk=(1, 3))
                # loss.update(cur_loss.data[0])
                loss.update(cur_loss.data.item())
                top1.update(cur_acc1[0].item())
                top5.update(cur_acc5[0].item())
                if iters % 100 == 0:
                    print('  loss:{}, acc1:{}, acc5:{}'.format(cur_loss.data.item(), cur_acc1[0].item(),
                                                               cur_acc5[0].item()))

            # Summary Writing
            self.summary_writer.add_scalar("epoch-loss", loss.avg, cur_epoch)
            self.summary_writer.add_scalar("epoch-top-1-acc", top1.avg, cur_epoch)
            self.summary_writer.add_scalar("epoch-top-5-acc", top5.avg, cur_epoch)

            # Print in console
            tqdm_batch.close()
            print("Epoch-" + str(cur_epoch) + " | " + "loss: " + str(
                loss.avg)[:7] + " - acc-top1: " + str(
                top1.avg)[:7] + "- acc-top5: " + str(top5.avg)[:7])

            # Checkpointing
            is_best = top1.avg > self.best_top1
            self.best_top1 = max(top1.avg, self.best_top1)
            self.save_checkpoint({
                'epoch': cur_epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_top1': self.best_top1,
                # 'mean': self.mean,
                # 'std': self.std,
                'optimizer': self.optimizer.state_dict(),
            }, train_is_best=is_best)

            # Evaluate on Validation Set
            if (cur_epoch+1) % self.args.test_every == 0 and self.valloader:
                self.test(self.valloader, cur_epoch)

    def test(self, testloader, cur_epoch=-1):
        loss, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker()

        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.eval()

        target_pre = []
        target_label = []
        target_cre = []
        target_error = []
        T = 0
        for data, target in testloader:

            if self.args.cuda:
                data, target = data.cuda(device=self.args.cuda_select, non_blocking=self.args.async_loading), \
                               target.cuda(device=self.args.cuda_select, non_blocking=self.args.async_loading)
            # data_var, target_var = Variable(data, volatile=True), Variable(target, volatile=True)
            with torch.no_grad():
                data_var, target_var = Variable(data), Variable(target)

            # Forward pass
            start = time.time()
            output = self.model(data_var)
            end = time.time()
            Time = end - start
            cur_loss = self.loss(output, target_var)
            T = T + Time
            # Top-1 and Top-5 Accuracy Calculation
            cur_acc1, cur_acc5 = self.compute_accuracy(output, target, topk=(1, 3))
            # loss.update(cur_loss.data[0])
            loss.update(cur_loss.data.item())
            top1.update(cur_acc1[0].item())
            top5.update(cur_acc5[0].item())

            if self.args.TestMode:
                Pos, label, Cre, Error = self.compute_Pos_Cre(F.softmax(output, dim=1), target)
                target_pre.extend(Pos)
                target_label.extend(label)
                target_cre.extend(Cre)
                target_error.extend(Error)

                # print('\nReal:', label,
                #       '\nPrediction:', Pos,
                #       '\nCredit:', Cre,
                #       '\nError:', Error,
                #       '\nTime:', Time
                #       )

        # plt.imshow(data[0].cpu().squeeze(), cmap='gray')
        #
        #         plt.title('Real:%i,Pred:%i,Credit:%f' % (label[0], Pos[0], Cre[0]))
        #         plt.plot([label[0], label[0]],
        #                  [0, data[0].size(1) - 1], color='green', linewidth=1.0)
        #
        #         plt.plot([Pos[0], Pos[0]],
        #                  [0, data[0].size(1) - 1], color='red', linewidth=1.0)
        #         plt.xticks(())
        #         plt.yticks(())
        #         plt.show()
        # AVgT = T / (100 / self.batch_size)
        # print("AVgT=\n", AVgT)

        if self.args.TestMode:
            self.save_comparisions(loss.avg, top1.avg, top5.avg, target_pre, target_label, target_cre, target_error,
                                   filename='comparisions')
        else:
            # Checkpointing
            test_is_best = top1.avg > self.test_best_top1
            self.test_best_top1 = max(top1.avg, self.test_best_top1)

            if test_is_best:
                self.save_checkpoint({
                    'epoch': cur_epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_top1': self.best_top1,
                    # 'mean': self.mean,
                    # 'std': self.std,
                    'optimizer': self.optimizer.state_dict(),
                }, test_is_best=test_is_best)
        if cur_epoch != -1 and not self.args.TestMode:
            # Summary Writing
            self.summary_writer.add_scalar("test-loss", loss.avg, cur_epoch)
            self.summary_writer.add_scalar("test-top-1-acc", top1.avg, cur_epoch)
            self.summary_writer.add_scalar("test-top-5-acc", top5.avg, cur_epoch)

        print("Test Results" + " | " + "loss: " + str(loss.avg)[:7] + " - acc-top1: " + str(
            top1.avg)[:7] + "- acc-top5: " + str(top5.avg)[:7])

    def save_checkpoint(self, state, train_is_best=False, test_is_best=False, filename='checkpoint.pth.tar'):
        if not test_is_best:
            torch.save(state, self.args.checkpoint_dir + filename)
            if train_is_best:
                shutil.copyfile(self.args.checkpoint_dir + filename,
                                self.args.checkpoint_dir + 'model_train_best.pth.tar')
        else:
            shutil.copyfile(self.args.checkpoint_dir + filename,
                            self.args.checkpoint_dir + 'model_test_best.pth.tar')

    def compute_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, idx = output.topk(maxk, 1, True, True)
        idx = idx.t()
        correct = idx.eq(target.view(1, -1).expand_as(idx))

        acc_arr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc_arr.append(correct_k.mul_(1.0 / batch_size))
        return acc_arr

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    def create_optimization(self):
        self.loss = nn.CrossEntropyLoss()

        if self.args.cuda:
            self.loss.cuda(self.args.cuda_select)

        self.optimizer = RMSprop(self.model.parameters(), self.args.learning_rate,
                                 momentum=self.args.momentum,
                                 weight_decay=self.args.weight_decay)

    def load_pretrained_model(self):
        try:
            print("Loading ImageNet pretrained weights...")
            pretrained_dict = torch.load(self.args.pretrained_path)
            self.model.load_state_dict(pretrained_dict)
            print("ImageNet pretrained weights loaded successfully.\n")
        except:
            print("No ImageNet pretrained weights exist. Skipping...\n")

    def load_checkpoint(self, filename):
        filename = self.args.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            # self.mean = checkpoint['mean']
            # self.std = checkpoint['std']
            self.start_epoch = checkpoint['epoch']
            self.best_top1 = checkpoint['best_top1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch']))
        except:
            print("No checkpoint exists from '{}'. Skipping...\n".format(filename))
            if self.args.TestMode:
                assert not self.args.TestMode, "No checkpoint exists from '{}'\n".format(filename)

    def compute_Pos_Cre(self, output, target):

        Pos = torch.max(output, 1)[1].view(1, -1)
        Cre = torch.max(output, 1)[0].view(1, -1)
        Error = Pos - target.view(1, -1)

        return Pos.cpu().detach().numpy().reshape(-1).tolist(), \
               target.cpu().detach().numpy().reshape(-1).tolist(), \
               Cre.cpu().detach().numpy().reshape(-1).tolist(), \
               Error.cpu().detach().numpy().reshape(-1).tolist()

    def save_comparisions(self, loss, acc_top1, acc_top5, target_pre, target_label, target_cre, target_error,
                          filename='comparisions'):

        filename = self.args.checkpoint_dir + filename + '_' + time.strftime('%Y%m%d_%H%M',
                                                                             time.localtime(time.time())) + '.txt'

        file = open(filename, 'w')
        file.write('Train epoch: %d\n' % self.start_epoch)
        file.write('Train best top1: %.4f\n' % self.best_top1)
        file.write('Test average loss: %.4f\n' % loss)
        file.write('Test accuracy top1: %.4f\n' % acc_top1)
        file.write('Test accuracy top5: %.4f\n' % acc_top5)
        file.write('Prediction:\n')
        file.write(str(target_pre))
        file.write('\nLabel:\n')
        file.write(str(target_label))
        file.write('\nCredit:\n')
        file.write(str(target_cre))
        file.write('\nError:\n')
        file.write(str(target_error))
        file.close()

        print('\nSave comparisions.\n')
