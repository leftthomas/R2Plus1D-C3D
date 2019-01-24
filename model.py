import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


class Model(nn.Module):
    def __init__(self, num_features, num_classes, num_iterations=3):
        super(Model, self).__init__()

        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.classifier = CapsuleLinear(out_capsules=num_classes, in_length=128, out_length=32, in_capsules=None,
                                        share_weight=True, routing_type='k_means', num_iterations=num_iterations)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = scatter_mean(x, data.batch, dim=0)

        return F.sigmoid(x)
