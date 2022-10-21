# ==========================================================
# Author: Tomas Jakab
# ==========================================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def get_marker_style(i, cmap='Dark2'):
  cmap = plt.get_cmap(cmap)
  colors = [cmap(c) for c in np.linspace(0., 1., 8)]
  markers = ['X', 'v', 'o', 's', 'd', '^', 'x', '+', 'p', 'P', '*']
  max_i = len(colors) * len(markers) - 1
  if i > max_i:
    raise ValueError('Exceeded maximum (' + str(max_i) + ') index for styles.')
  c = i % len(colors)
  m = int(i / len(colors))
  return colors[c], markers[m]


def single_marker_style(color, marker):
  return lambda _: (color, marker)


def plot_landmark(ax, landmark, k, size=1.5, zorder=2, cmap='Dark2',
                  style_fn=None):
  if style_fn is None:
    c, m = get_marker_style(k, cmap=cmap)
  else:
    c, m = style_fn(k)
  ax.scatter(landmark[0], landmark[1], c=c, marker=m,
             s=(size * mpl.rcParams['lines.markersize']) ** 2,
             zorder=zorder)


def plot_landmarks(ax, landmarks, size=1.5, zorder=2, cmap='Dark2', style_fn=None):
  for k, landmark in enumerate(landmarks):
    plot_landmark(ax, landmark, k, size=size, zorder=zorder,
                  cmap=cmap, style_fn=style_fn)

# =================
# unfold_heatmap

def unfold_heatmap(hm, pics_per_line=23):
    """[summary]

    Args:
        hm (tensor): NxHxW
    """
    B,C,H,W = hm.shape

    # gen the whole figure
    x_num = min(pics_per_line, C)
    y_num = C // x_num + int((C % x_num) > 0)
    fig = np.zeros((y_num*H, x_num*W))

    # plot each hm
    for idx, hmi in enumerate(hm.detach().cpu().squeeze().numpy()):
        pos_x = idx % x_num
        pos_y = idx // x_num
        # print(pos_x, pos_y, x_num, y_num, fig[pos_y*H:(pos_y+1)*H, pos_x*W:(pos_x+1)*W].shape)
        fig[pos_y*H:(pos_y+1)*H, pos_x*W:(pos_x+1)*W] = hmi

    # if C == 5 or C == 10:
    #     hm_pred_show = np.concatenate(hm.squeeze().numpy(), 1)
    # elif C == 68:
    #     hm_pred_show = np.concatenate(hm.squeeze().numpy(), 1).reshape((H*4, -1))
    # else:
    #     raise NotImplementedError('error for C = %d, write it now!' % C)
    return fig
