import numpy as np

def get_default_txeture(faces_shape, t_size=3):
    default_tex = np.ones((1, faces_shape, t_size, t_size, t_size, 3))
    blue = np.array([156, 199, 234.]) / 255.
    default_tex = default_tex * blue
    return default_tex