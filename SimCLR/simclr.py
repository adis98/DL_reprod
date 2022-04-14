import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            # counter = np.zeros(10) UNCOMMENT FOR PCA/TSNE
            # new_features = None
            # new_labels = None
            # colors = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
            for images, labels in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                # for lab in labels:           UNCOMMENT THESE LINES FOR PCA/TSNE
                #     if(counter[lab] < 100):
                #         counter[lab] +=1
                images = images.to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    # with torch.no_grad(): UNCOMMENT THIS FOR PCA/TSNE
                        features = self.model(images)
                        # labels_mod = torch.cat((labels,labels),dim=0) UNCOMMENT THIS LINE FOR PCA/TSNE

                        # if(new_features is not None): UNCOMMENT THESE LINES FOR PCA/TSNE
                        #     new_features = torch.cat((new_features,features),dim=0)
                        #     new_labels = torch.cat((new_labels,labels_mod),dim=0)
                        # else:
                        #     new_features = features
                        #     new_labels = labels_mod

                        logits, labels = self.info_nce_loss(features)
                        loss = self.criterion(logits, labels)
                # if(np.sum(counter) == 1000): //UNCOMMENT THESE LINES FOR PCA/TSNE
                #     # fig = plt.figure()
                #     ax = plt.axes(projection='3d')
                #     tsne_or_pca = PCA(n_components=3) #this can be changed to PCA or TSNE depending on what you want
                #     normalized_features = new_features.cpu().numpy()
                #     principalComponents = tsne_or_pca.fit_transform(normalized_features)
                #     principalComps = []
                #     for color in colors:
                #         for i in range(len(new_labels)):
                #             if(new_labels[i] == colors.index(color)):
                #                 principalComps.append(principalComponents[i])
                #         principalComps = np.array(principalComps)
                #
                #         if(len(principalComps) != 0):
                #             x_arr  = np.array(principalComps[:,0])
                #             y_arr = np.array(principalComps[:,1])
                #             z_arr = np.array(principalComps[:,2])
                #             ax.scatter3D(x_arr,y_arr,z_arr,c = color)
                #         principalComps = []
                #     plt.show()
                #     break

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints

        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
