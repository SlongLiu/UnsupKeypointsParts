import torch
import torch.nn as nn

import neural_renderer
from geoutils import geom_utils


class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, img_size=256):
        super(NeuralRenderer, self).__init__()
        self.renderer = neural_renderer.Renderer()

        # Adjust the core renderer
        self.renderer.image_size = img_size
        self.renderer.perspective = False

        # Set a default camera to be at (0, 0, -2.732)
        self.renderer.eye = [0, 0, -2.732]

        # Make it a bit brighter for vis
        self.renderer.light_intensity_ambient = 0.8

        self.proj_fn = geom_utils.orthographic_proj_withz
        self.offset_z = 5.

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.light_intensity_ambient = 1
        self.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        self.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, cams, textures=None):
        # print('vertices.shape:', vertices.shape, 'cams.shape:', cams.shape)
        verts = self.proj_fn(vertices, cams, offset_z=self.offset_z)
        vs = verts.clone()
        vs[:, :, 1] *= -1
        fs = faces.clone()
        if textures is None:
            self.mask_only = True
            masks = self.renderer.render_silhouettes(vs, fs)
            return masks
        else:
            self.mask_only = False
            ts = textures.clone()
            imgs = self.renderer.render(vs, fs, ts)[0] #only keep rgb, no alpha and depth
            return imgs