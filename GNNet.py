import os
package_directory = os.path.dirname(os.path.abspath(__file__))
directory_TGM = os.path.join(package_directory, 'torch_geometric')
print(directory_TGM)

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dotdict
import importlib.util
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import Graph
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
import csv



args = dotdict({
    'lr': 0.01,
    'epochs': 15,
    'batch_size': 8,
    'num_channels': 256,
    'l2_coeff':1e-4
})


class GNNetWrapper():
    def __init__(self,argsargs=None,save_info=False):
        self.nnet = CustomGNN(num_features=1,channels=args.num_channels)

        # save info ot the results file
        if save_info:
            with open(argsargs.resultsFilePath,"a",newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                learning_rate = "Learning rate: " + str(args.lr)
                epochs = "Epochs: " + str(args.epochs)
                batch_size = "Batch size: " + str(args.batch_size)
                num_channels = "Number of channels: " + str(args.num_channels)
                l2_coeff = "L2 coefficient: " + str(args.l2_coeff)
                csv_writer.writerow([learning_rate, epochs, batch_size, num_channels, l2_coeff])

            
    
    def train(self, examples):
        """
        examples: list of examples, each example is of form (graph, pi, v)
        """
        #device = torch.device("mps")
        input_graphs, target_pis, target_values = list(zip(*examples))
        
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr)
        
        train_dataset = CustomGraphDataset(input_graphs,target_pis,target_values)
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=custom_collate)
        device = torch.device("cpu")
        self.nnet.to(device)

        def custom_loss(pi,target_pi,value,target_value):
            print("pred_pi shape:", pi.shape)
            print("target_pi shape:", target_pi.shape)
            print("pred_value shape:", value.shape)
            print("target_value shape:", target_value.shape)

            pi = pi.view_as(target_pi)
            #print("pred_pi shape:", pi.shape)
            mse_loss = nn.MSELoss()(value.view(-1),target_value.view(-1))
            cross_entropy_loss = nn.CrossEntropyLoss()(pi,target_pi)
            #bce_loss = nn.BCELoss()(pi,target_pi)
            #kl_div_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(pi,dim=1),target_pi)

            # L2 regularization
            l2_reg = torch.tensor(0., device=value.device)
            for param in self.nnet.parameters():
                l2_reg += torch.norm(param,p=2)**2
            l2_loss = args.l2_coeff * l2_reg

            return mse_loss, cross_entropy_loss, l2_loss

        for epoch in tqdm(range(args.epochs),desc="Training GNNet"):
            self.nnet.train()
            total_loss = 0
            total_value_loss = 0
            total_policy_loss = 0
            total_reg_loss = 0
            #for i in range(len(input_graphs)):
            for graph,target_pi,target_value in train_loader:
                # train per example => batch could be better
                # graph = input_graphs[i]
                # target_pi = torch.tensor(target_pis[i])
                # target_value = torch.tensor(target_values[i])
                # print("Graph: ", graph)
                # print("Target pi: ", target_pi.shape, "\ntype: ", type(target_pi))
                # print("Target value: ", target_value.shape,"\ntype: ", type(target_value))

                # graph = graph.to(device)
                # target_pi = target_pi.to(device)
                # target_value = target_value.to(device)

                optimizer.zero_grad()
                pred_pi, pred_value = self.nnet(graph)
                print("Pred pi: ", len(pred_pi), "\ntarget pi: ", len(target_pi))
                print("Pred value: ", pred_value, "\ntarget value: ", target_value)

                value_loss , policy_loss, reg_loss = custom_loss(pred_pi,target_pi,pred_value,target_value)
                loss = value_loss + policy_loss + reg_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                total_reg_loss += reg_loss.item()
                
            print("Epoch: {}, Avg loss: {:.5f}, Avg value loss: {:.5f}, Avg policy loss: {:.5f},Avg reg. loss{:.5f},Examples: {}".format(epoch+1, total_loss/(len(input_graphs)),total_value_loss/(len(input_graphs)),total_policy_loss/(len(input_graphs)),total_reg_loss/len(input_graphs),len(input_graphs)))
        self.nnet.to("cpu")            



    def predict(self, state):
        # turn board representation into a graph
        # use the graph as input to the model

        data = Graph.state_to_graph_data(state)
        self.nnet.eval()
        with torch.no_grad():
            edge_probs,value = self.nnet(data)
        
        #print("Edge probs: ", edge_probs)
        #pi = Graph.edges_to_actions(self.game,edge_probs.tolist())
        #print("Pi: ", pi)

        # have to check if the actions in pi are legal 
        # only return policy for legal states
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
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(num_features, channels),nn.ReLU(),nn.LayerNorm(channels)))
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)),edge_dim=1) # edge_dim = 40

        # Fully-connected layers with batch normalization, ReLU activation, and dropout
        self.fc1 = nn.Sequential(nn.Linear(3* channels + num_features , 2 * channels),nn.ReLU(),nn.BatchNorm1d(2*channels))
        self.fc2 = nn.Sequential(nn.Linear(2*channels, channels),nn.ReLU(),nn.BatchNorm1d(channels))

        # Policy head that outputs a probability value for each edge in the graph
        self.policy_head = nn.Sequential(nn.Linear(channels, 1),nn.Sigmoid())

        # Value head
        self.value_head = nn.Sequential(nn.Linear(channels, 1),nn.Tanh())

    def forward(self, data):
        x, edge_index, edge_attr , batch = data.x, data.edge_index, data.edge_attr, data.batch
        # print("x: ", x.shape)
        # print(x.dtype)
        # print("edge index: ", edge_index.shape)
        # print("edge attr: ", edge_attr.shape)
        # print(edge_attr.dtype)

        # !!!!! edge_attr is not used
        x1 = self.conv1(x, edge_index, edge_attr)
        x2 = self.conv2(x1, edge_index,edge_attr)
        x3 = self.conv3(x2, edge_index,edge_attr)

        # Concatenate intermediate representations
        x_concat = torch.cat([x, x1, x2, x3], dim=-1)
        x_layer_1 = self.fc1(x_concat)
        x_layer_2 = self.fc2(x_layer_1)

        # Compute policy and value
        edge_probs = self.policy_head(x_layer_2[edge_index[0]]).squeeze()
        value = self.value_head(global_mean_pool(x_layer_2,batch)).squeeze()

        return edge_probs, value


class CustomGraphDataset(Dataset):
    def __init__(self, input_graphs, target_pis, target_values):
        self.input_graphs = input_graphs
        self.target_pis = target_pis
        self.target_values = target_values

    def __getitem__(self, index):
        device = torch.device("cpu")
        return self.input_graphs[index].to(device), torch.tensor(self.target_pis[index]).to(device), torch.tensor(self.target_values[index]).to(device)

    def __len__(self):
        return len(self.input_graphs)
    
def custom_collate(batch):
    graphs, target_pis, target_values = zip(*batch)
    return graphs, torch.tensor(target_pis), torch.tensor(target_values)

def create_batch(input_graphs):
    batch_graph = Batch.from_data_list(input_graphs)
    return batch_graph



class CustomGINConv(MessagePassing):
    def __init__(self, nn, **kwargs):
        super(CustomGINConv, self).__init__(aggr='add', **kwargs)  # "Add" aggregation
        self.nn = nn

    def forward(self, x, edge_index, edge_attr):
        edge_attr = edge_attr.view(-1, 1)  # Reshape to have a second dimension

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return aggr_out
