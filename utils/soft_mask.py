import torch


def get_smooth_step(n, b):
    x = torch.linspace(-1, 1, n)
    y = 0.5 + 0.5 * torch.tanh(x / b)
    return y

def get_smooth_mask(h, w, margin, step):
    b = 0.4
    step_up = get_smooth_step(step, b)
    step_down = get_smooth_step(step, -b)

    def create_strip(size):
        return torch.cat(
            [torch.zeros(margin, dtype=torch.float32),
            step_up,
            torch.ones(size - 2 * margin - 2 * step, dtype=torch.float32),
            step_down,
            torch.zeros(margin, dtype=torch.float32)], axis=0)

    mask_x = create_strip(w)
    mask_y = create_strip(h)
    mask2d = mask_y[:, None] * mask_x[None]
    return mask2d # H,W