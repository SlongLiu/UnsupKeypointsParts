import torch
import torch.nn as nn

from utils.utils import gifdict
from geoutils.laplacian import LaplacianModule

from .baseloss import BaseLoss
    

class keyPointsLoss(BaseLoss):
    def __init__(self, item_map=dict(x='x', y='y'), **kw):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.item_map = item_map

    def run(self, x, y):
        return self.criterion(x, y)


class EntropyLoss(BaseLoss):
    def __init__(self, item_map=dict(x='x'), **kw):
        super().__init__()
        self.item_map = item_map

    def run(self, x):
        entropy = -torch.sum(x * torch.log(x), 1)
        return entropy.mean()


class L2NormLoss(BaseLoss):
    def __init__(self, item_map=dict(x='x'), **kw):
        super().__init__()
        self.item_map = item_map

    def run(self, x):
        x = x.view(-1, x.size(2))
        return torch.mean(torch.norm(x, p=2, dim=1))


class LaplacianLoss(BaseLoss):
    """
    Encourages minimal mean curvature shapes.
    """
    def __init__(self, faces=None, item_map=dict(verts='verts'), **kw):
        super().__init__()
        # Input:
        #  faces: B x F x 3
        # V x V
        # self.laplacian = LaplacianModule(faces)
        self.laplacian = None
        self.Lx = None
        self.item_map = item_map

    def get_face(self, faces):
        self.laplacian = LaplacianModule(faces)

    def run(self, verts):
        self.Lx = self.laplacian(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = torch.norm(Lx, p=2, dim=1).mean()
        return loss

    def forward(self, ddata):
        if self.laplacian is None:
            self.get_face(ddata['output']['faces'])
        return self.run(**{k:gifdict(ddata, v) for k,v in self.item_map.items()})

    # def visualize(self, verts, mv=None):
    #     # Visualizes the laplacian.
    #     # Verts is B x N x 3 Variable
    #     Lx = self.Lx[0].data.cpu().numpy()

    #     V = verts[0].data.cpu().numpy()

    #     from psbody.mesh import Mesh
    #     F = self.laplacian.F_np[0]
    #     mesh = Mesh(V, F)

    #     weights = np.linalg.norm(Lx, axis=1)
    #     mesh.set_vertex_colors_from_weights(weights)

    #     if mv is not None:
    #         mv.set_dynamic_meshes([mesh])
    #     else:
    #         mesh.show()
    #         import ipdb; ipdb.set_trace()
        