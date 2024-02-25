import os
import importlib.util
from utils import dotdict
import sys
import math
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from torch.optim import lr_scheduler

import Graph



class GNNetWrapper():
    def __init__(self,args,save_info=False):
        self.nnet = CustomGNN(num_features=2,channels=args.num_channels)
        self.args = args
        
        # save the model info to a csv file
        if args.saveResults:
            with open(self.args.resultsFilePath,"a",newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                learning_rate = "Learning rate: " + str(args.lr)
                epochs = "Epochs: " + str(self.args.epochs)
                batch_size = "Batch size: " + str(self.args.batch_size)
                num_channels = "Number of channels: " + str(self.args.num_channels)
                l2_coeff = "L2 coefficient: " + str(args.l2_coeff)
                temp = "Temperature threshold: " + str(self.args.tempThreshold)
                csv_writer.writerow([learning_rate, epochs, batch_size, num_channels, l2_coeff,temp])

    def custom_loss(self,pi,target_pi,value,target_value):

        pi = target_pi.view_as(target_pi)
        # MSE loss for values
        mse_loss = nn.MSELoss()(value.view(-1),target_value.view(-1))
        # Cross entropy loss for policy
        cross_entropy_loss = nn.CrossEntropyLoss()(pi,target_pi)
        # L2 regularization
        l2_reg = torch.tensor(0., device=value.device)
        for param in self.nnet.parameters():
            l2_reg += torch.norm(param,p=2)**2
        l2_loss = self.args.l2_coeff * l2_reg

        return mse_loss, cross_entropy_loss, l2_loss
    
    def train(self, examples):
        """
        examples: list of examples, each example is of form (graph, pi, v)
        """

        #device = torch.device('mps')
 
        input_graphs, target_pis, target_values = list(zip(*examples))
    
        
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        # Learning rate triangular schedule
        #scheduler = CyclicalLR(optimizer, schedule=lambda epoch,lr: triangular_schedule(epoch, 0.0001, 0.0005, 7))

        train_dataset = CustomGraphDataset(input_graphs,target_pis,target_values)
        train_loader = DataLoader(train_dataset,batch_size=self.args.batch_size,shuffle=True,collate_fn=custom_collate)

        for epoch in tqdm(range(self.args.epochs),desc="Training GNNet"):
            self.nnet.train()
            total_loss = 0
            total_value_loss = 0
            total_policy_loss = 0
            total_reg_loss = 0
            for graph,target_pi,target_value in train_loader:

                optimizer.zero_grad()
                pred_pi, pred_value = self.nnet(graph)

                value_loss , policy_loss, l2_loss = self.custom_loss(pred_pi,target_pi,pred_value,target_value)
                loss = value_loss + policy_loss + l2_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                total_reg_loss += l2_loss.item()
            #scheduler.step()
            print("Epoch: {}, Avg loss: {:.5f}, Avg value loss: {:.5f}, Avg policy loss: {:.5f},Avg reg. loss: {:.5f},Examples: {}".format(epoch+1, total_loss/(len(input_graphs)),total_value_loss/(len(input_graphs)),total_policy_loss/(len(input_graphs)),total_reg_loss/len(input_graphs),len(input_graphs)))

    def predict(self, state):
        
        # Convert state to graph data
        data = Graph.state_to_graph_data(state)

        self.nnet.eval()
        with torch.no_grad():
            edge_probs,value = self.nnet(data)

        return edge_probs,value

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(self.nnet.state_dict(), filepath)
        #self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        
        filepath = os.path.join(folder, filename)
        loaded_state_dict = torch.load(filepath)
        self.nnet.load_state_dict(loaded_state_dict)
        #self.nnet.model.load_weights(filepath)


    

class CustomGNN(torch.nn.Module):
    # num_features: number of features per node = 2
    def __init__(self, num_features, channels):
        super(CustomGNN, self).__init__()

        # GINConv layers with layer normalization and ReLU activation
        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, channels),nn.ReLU(),nn.LayerNorm(channels)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)))

        # Fully-connected layers with batch normalization, ReLU activation, and dropout
        self.fc1 = nn.Sequential(nn.Linear(3 * channels,channels),nn.ReLU(),nn.BatchNorm1d(channels),nn.Dropout(0.5)) 
        self.fc2 = nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.BatchNorm1d(channels),nn.Dropout(0.5)) 

        # Policy head that outputs a probability value for each edge in the graph
        self.policy_head = nn.Sequential(nn.Linear(channels, 1),nn.Sigmoid())

        # Value head
        self.value_head = nn.Sequential(nn.Linear(channels, 1),nn.Tanh())

    def forward(self, data):
        x, edge_index, edge_attr , batch = data.x, data.edge_index, data.edge_attr, data.batch

        x1 = F.relu(self.conv1(x, edge_index)) 
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))

        # Concatenate intermediate representations
        x_concat = torch.cat([x1, x2, x3], dim=-1)
        x_fc = self.fc2(self.fc1(x_concat))

        # Compute policy and value
        edge_probs = self.policy_head(x_fc[edge_index[0]]).squeeze()
        value = self.value_head(global_mean_pool(x_fc,batch)).squeeze()

        return edge_probs, value

class CustomGNNEdges(torch.nn.Module):
    # num_features: number of features per node = 2
    def __init__(self, num_features, channels):
        super(CustomGNN, self).__init__()

        # GINConv layers with layer normalization and ReLU activation
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(num_features, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1) # edge_dim = 40
        self.conv4 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1) # edge_dim = 40
        self.conv5 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1) # edge_dim = 40
        self.conv6 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1) # edge_dim = 40
        self.conv7 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1) # edge_dim = 40
        self.conv8 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1) # edge_dim = 40
        self.conv9 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1) # edge_dim = 40


        # Fully-connected layers with batch normalization, ReLU activation, and dropout
        self.fc1 = nn.Sequential(nn.Linear(9* channels + num_features , 2 * channels),nn.ReLU(),nn.BatchNorm1d(2*channels))
        self.fc2 = nn.Sequential(nn.Linear(2*channels, channels),nn.ReLU(),nn.BatchNorm1d(channels))

        # Policy head that outputs a probability value for each edge in the graph
        self.policy_head = nn.Sequential(nn.Linear(channels, 1),nn.Sigmoid())

        # Value head
        self.value_head = nn.Sequential(nn.Linear(channels, 1),nn.Tanh())

    def forward(self, data):
        x, edge_index, edge_attr , batch = data.x, data.edge_index, data.edge_attr, data.batch
    
        x1 = self.conv1(x, edge_index, edge_attr)
        x2 = self.conv2(x1, edge_index,edge_attr)
        x3 = self.conv3(x2, edge_index,edge_attr)
        x4 = self.conv4(x3, edge_index,edge_attr)
        x5 = self.conv5(x4, edge_index,edge_attr)
        x6 = self.conv6(x5, edge_index,edge_attr)
        x7 = self.conv7(x6, edge_index,edge_attr)
        x8 = self.conv8(x7, edge_index,edge_attr)
        x9 = self.conv9(x8, edge_index,edge_attr)


        # Concatenate intermediate representations
        x_concat = torch.cat([x, x1, x2, x3,x4,x5,x6,x7,x8,x9], dim=-1)
        x_layer_1 = self.fc1(x_concat)
        x_layer_2 = self.fc2(x_layer_1)

        # Compute policy and value
        edge_probs = self.policy_head(x_layer_2[edge_index[0]]).squeeze()
        value = self.value_head(global_mean_pool(x_layer_2,batch)).squeeze()

        return edge_probs, value

# custom class to handle batching of graphs
class CustomGraphDataset(Dataset):
    def __init__(self, input_graphs, target_pis, target_values):
        self.input_graphs = input_graphs
        self.target_pis = target_pis
        self.target_values = target_values

    def __getitem__(self, index):
        return self.input_graphs[index], torch.tensor(self.target_pis[index]), torch.tensor(self.target_values[index])

    def __len__(self):
        return len(self.input_graphs)
    
def custom_collate(batch):
    input_graphs, target_pis, target_values = zip(*batch)
    batch_graph = Batch.from_data_list(input_graphs)
    batch_target_pis = torch.tensor(target_pis, dtype=torch.float)
    batch_target_values = torch.tensor(target_values, dtype=torch.float)
    return batch_graph, batch_target_pis, batch_target_values

def create_batch(input_graphs):
    batch_graph = Batch.from_data_list(input_graphs)
    return batch_graph

# cyclical learning rate scheduler
class CyclicalLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
# triangular learning rate schedule
def triangular_schedule(epoch, base_lr, max_lr, step_size):
    cycle = math.floor(1 + epoch/(2*step_size))
    x = abs(epoch/step_size - 2*cycle + 1)
    lr = base_lr + (max_lr - base_lr) * max(0, (1-x))
    return lr