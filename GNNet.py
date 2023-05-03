import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import dotdict
from torch_geometric.nn import GINConv, global_mean_pool
import os
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Dataset
import Graph

args = dotdict({
    'lr': 0.0001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 512,
})


class GNNetWrapper():
    def __init__(self, game):
        self.nnet = CustomGNN(num_features=2,channels=512)
        self.game = game
    

    def train(self, examples):
        """
        examples: list of examples, each example is of form (graph, pi, v)
        """

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = self.nnet.to(device)
        # data = examples[0].to(device)

        # turn examples into numpy arrays
        
        input_graphs, target_pis, target_values = list(zip(*examples))
        #print("input_graphs type:", type(input_graphs))
        #print("target_pis type:", type(target_pis))
        #print("target_values type:", type(target_values))
        
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr)
        criterion = nn.BCELoss()
        # train_dataset = CustomGraphDataset(input_graphs,target_pis,target_values)
        # train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=custom_collate)

        # def combined_loss(edge_probs, target_edge_probs, value_preds, target_values, alpha=0.5):
        #     print(edge_probs)
        #     edge_probs = torch.tensor(edge_probs)
        #     print(target_edge_probs)
        #     target_edge_probs = torch.tensor(target_edge_probs)
        #     value_preds = torch.tensor(value_preds)
        #     target_values = torch.tensor(target_values)
        #     print("edge_probs shape:", edge_probs.shape)
        #     print("target_edge_probs shape:", target_edge_probs.shape)
        #     print("value_preds shape:", value_preds.shape)
        #     print("target_values shape:", target_values.shape)
        #     edge_loss = criterion(edge_probs, target_edge_probs)
        #     value_loss = criterion(value_preds, target_values)
        #     return alpha * edge_loss + (1 - alpha) * value_loss
        
        # total_loss = 0
        # for epoch in range(args.epochs):
        #     self.nnet.train()
        #     for batch_graph,target_pis,target_values in train_loader:
        #         optimizer.zero_grad()
        #         edge_probs, value = self.nnet(batch_graph)
        #         #pred_pis = Graph.edges_to_actions(self.game,edge_probs)
        #         loss = combined_loss(edge_probs, target_pis, value, target_values)
        #         loss.backward()
        #         optimizer.step()
        #         total_loss += loss.item()
        #         print("Epoch: {}, Loss: {:.4f}".format(epoch, total_loss))

        def custom_loss(pi,target_pi,value,target_value):
            mse_loss = nn.MSELoss()(value,target_value)
            cross_entropy_loss = nn.CrossEntropyLoss()(pi,target_pi)
            return mse_loss, cross_entropy_loss

        for epoch in range(args.epochs):
            self.nnet.train()
            total_loss = 0
            total_value_loss = 0
            total_policy_loss = 0
            for i in range(len(input_graphs)):
                # train per example => batch could be better
                graph = input_graphs[i]
                target_pi = torch.tensor(target_pis[i])
                target_value = torch.tensor(target_values[i])
                #print("Graph: ", graph)
                #print("Target pi: ", target_pi, "\ntype: ", type(target_pi))
                #print("Target value: ", target_value,"\ntype: ", type(target_value))

                optimizer.zero_grad()
                pred_pi, pred_value = self.nnet(graph)

                # print("Pred pi: ", len(pred_pi), "\ntarget pi: ", len(target_pi))
                # print("Pred value: ", pred_value, "\ntarget value: ", target_value)
                #loss = custom_loss(pred_pi,target_pi,pred_value,target_value)
                value_loss , policy_loss = custom_loss(pred_pi,target_pi,pred_value,target_value)
                loss = value_loss + policy_loss
                #loss = F.nll_loss(pred_pi,target_pi) + F.mse_loss(pred_value,target_value)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
            print("Epoch: {}, Total loss: {:.4f}, Total value loss: {:.2f}, Total policy loss: {:.2f},Examples: {}".format(epoch, total_loss,total_value_loss,total_policy_loss,len(input_graphs)))
            



    def predict(self, state):
        

        # turn board representation into a graph
        # use the graph as input to the model

        data = Graph.state_to_graph_data(state,self.game)
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
        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, channels),nn.ReLU(),nn.LayerNorm(channels)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(channels, channels),nn.ReLU(),nn.LayerNorm(channels)))

        # Fully-connected layers with batch normalization, ReLU activation, and dropout
        self.fc1 = nn.Sequential(nn.Linear(3 * channels, channels),nn.ReLU(),nn.BatchNorm1d(channels),nn.Dropout(0.5))
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


class CustomGraphDataset(Dataset):
    def __init__(self, input_graphs, target_pis, target_values):
        self.input_graphs = input_graphs
        self.target_pis = target_pis
        self.target_values = target_values

    def __getitem__(self, index):
        return self.input_graphs[index], self.target_pis[index], self.target_values[index]

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