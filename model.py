import torch
import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops

from utils import global_sort_pool


class Model(nn.Module):
    def __init__(self, num_features, num_classes, num_iterations=3):
        super(Model, self).__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.pool = CapsuleLinear(out_capsules=16, in_length=97, out_length=32, in_capsules=None,
                                  share_weight=True, routing_type='k_means', similarity='tonimoto',
                                  num_iterations=num_iterations)
        self.classifier_1 = Linear(512, 128)
        self.classifier_2 = Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)

        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = self.pool(global_sort_pool(x, batch))
        out = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(out))
        classes = F.log_softmax(self.classifier_2(out), dim=-1)

        return classes
