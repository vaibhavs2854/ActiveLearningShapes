import numpy as np
import torch

from torchvision import transforms
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import get_DataLoader
from viz import display_image_annotation

# import copy

class disc_model(): 
    def __init__(self):
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.loss_tracker = []
        
        self.dataloader = None
        return
    
    def load_model(self, dataloader): 
        num_classes=2

        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=False)
        self.model.classifier[6] = nn.Linear(4096,num_classes)
        self.model = self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCEWithLogitsLoss();
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9)
        self.loss_tracker = [] #plot loss
        self.dataloader = dataloader
        
        return
    
    def initialize_model(self, batch_size=32,epochs=15, randomize = True): 
        #Initial training on mismatched image-label pairs (half matched w/ label 1, half mismatched w/ label 0)
        for epoch in range(epochs):
            self.model.train()
            tr_loss = 0.0
            for image,mask,_ in tqdm(self.dataloader):
                feed_in_data = torch.empty((image.shape[0],3,256,256))
                labels = [1]*image.shape[0] 
                if randomize: # randomly show 0,1
                    for i in range(image.shape[0]):
                        if(i<image.shape[0]/2):
                            feed_in_data[i] = torch.stack([image[i],image[i],mask[i+1]]).squeeze()
                            labels[i] = 0;
                        else:
                            feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
                            labels[i] = 1; #1 is correct label, 0 is incorrect label
                images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
                images = images.cuda()
                labels = torch.from_numpy(np.array(labels)).cuda()
                self.optimizer.zero_grad()

                y = self.model(images)
                loss = self.criterion(y,labels)
                self.loss_tracker.append(loss.detach().cpu().item()/batch_size)
                loss.backward()
                self.optimizer.step()
    
    def update_model(self, oracle_results, batch_size=32, num_epochs=1):
        self.model.train()
        #Retrain one epoch
        for epoch in range(num_epochs):
            #make another dataloader w/ oracle results.
            for image,mask,patient_id in tqdm(self.dataloader):
                    feed_in_data = torch.empty((image.shape[0],3,256,256))
                    labels = [0]*image.shape[0] 
                    for i in range(image.shape[0]):
                        #print(batch_size/2)
                        if patient_id[i] in list(oracle_results.keys()) and oracle_results[patient_id[i]]!=2: #if currently in dict
                            feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
                            labels[i] = oracle_results[patient_id[i]]
                        else:
                            feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
                            labels[i] = 2       
                    #Normalize the image
                    images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
                    #Remove all indices that aren't an oracle
                    removed_indices = [i for i in range(image.shape[0]) if labels[i]==2]
                    dummy = torch.empty((image.shape[0] - len(removed_indices),3,256,256))
                    new_labels = [0]*(images.shape[0] - len(removed_indices))
                    cur_index = 0
                    for i in range(image.shape[0]):
                        if i in removed_indices:
                            continue
                        try:
                            dummy[cur_index] = images[i]
                            new_labels[cur_index] = labels[i]
                        except:
                            print("Error with removing indices")
                        cur_index+=1
                    #images = copy.deepcopy(dummy)
                    #labels = copy.deepcopy(new_labels)
                    dummy = dummy.cuda()
                    new_labels = torch.from_numpy(np.array(new_labels)).cuda()
                    self.optimizer.zero_grad()
                    if(dummy.shape[0]==0):
                        continue
                    y = self.model(dummy)

                    loss = self.criterion(y,new_labels)
                    self.loss_tracker.append(loss.detach().cpu().item()/batch_size)
                    loss.backward()
                    self.optimizer.step()
        self.model.eval()
        return 

    #evaluate, keep track of dict w/ (patient_id -> output of model) Sort by (|output-0.5|), take min and these are "unsure" classifications
    def get_scores(self, dataloader):
        patient_scores = {}
        self.model.eval()
        for image,mask,patient_id in tqdm(dataloader):
            feed_in_data = torch.empty((image.shape[0],3,256,256))
            for i in range(image.shape[0]):
                feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
            images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
            images = images.cuda()

            output = nn.Softmax(dim=1)(self.model(images))
            #print(type(output))
            #print(output.shape)
            for i in range(images.shape[0]):
                #print(output[i].shape)
                patient_scores[patient_id[i]] = output[i].cpu().detach()[1].item()

        patient_scores = {k: patient_scores[k] for k in sorted(patient_scores, key=patient_scores.get)}
        return patient_scores
    
    def plot_loss(self):
        plt.plot(self.loss_tracker)
        plt.xlabel('batches seen') 
        plt.ylabel(f'loss ({self.criterion})') 
        plt.title('Discriminator Training Loss')
        
        for i in range(int(len(self.loss_tracker)/len(self.dataloader))): 
            plt.axvline(x = (i+1) * len(self.dataloader), color = 'lightblue', label = f'epoch {i} complete')

        plt.show() 
        
    def plot_distribution(self, im_dir): 
        #get all images in im_dir
        im_dataloader = get_DataLoader(im_dir, self.dataloader.batch_size, 2)
        scores = []
        self.model.eval()
        
        for image,mask,_ in tqdm(im_dataloader):
            feed_in_data = torch.empty((image.shape[0],3,256,256))
            for i in range(image.shape[0]):
                feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
            images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
            images = images.cuda()

            output = nn.Softmax(dim=1)(self.model(images))
            for i in range(images.shape[0]):
                scores.append(output[i].cpu().detach()[1].item())
                
        sns.distplot(scores)
        
    def show_disc(self, im_dir):
        #get all images in im_dir
        im_dataloader = get_DataLoader(im_dir, self.dataloader.batch_size, 2)
        
        self.model.eval()
        
        for image,mask,_ in tqdm(im_dataloader):
            feed_in_data = torch.empty((image.shape[0],3,256,256))
            for i in range(image.shape[0]):
                feed_in_data[i] = torch.stack([image[i],image[i],mask[i]]).squeeze()
            images = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) (feed_in_data)
            images = images.cuda()
            
            output = nn.Softmax(dim=1)(self.model(images))
            
            img_pairs = []
            annotations = []
            for i in range(images.shape[0]):
                img_pairs.append([np.moveaxis(image[i].cpu().detach().numpy(),0,-1), 
                                  np.moveaxis(mask[i].cpu().detach().numpy(),0,-1)])
                annotations.append(f'score: {output[i].cpu().detach()[1].item():.3f}')
                
            plt.show(display_image_annotation(img_pairs,annotations))
            
            cont = input("See more? (y/n) ")
            if cont != 'y': 
                break
            
        