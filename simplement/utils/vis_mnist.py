def print_mnist_ascii(img, use_bg=True):
    """
    img: 2D numpy array or torch.Tensor with values in [0,1]
    use_bg: if True, color the background; else color the foreground text
    """
    ESC = "\x1b["
    RESET = ESC + "0m"
    H, W = img.shape

    for i in range(H):
        row_str = ""
        for j in range(W):
            v = float(img[i, j])
            # clamp just in case
            v = max(0.0, min(1.0, v))
            # map to 0–255
            color = int(v * 255)
            # choose a “block” character
            char = "  "  # two spaces look like a square cell
            # or use "█" for a tighter cell
            if use_bg:
                row_str += f"{ESC}48;5;{color}m{char}"
            else:
                row_str += f"{ESC}38;5;{color}m{char}"
        row_str += RESET  # reset at end of each line
        print(row_str)

def print_mnist_gray(img, use_bg=True):
    ESC = "\x1b["
    RESET = ESC + "0m"
    H, W = img.shape

    for i in range(H):
        row_str = ""
        for j in range(W):
            v = float(img[i, j])
            v = max(0.0, min(1.0, v))
            # map v∈[0,1] to codes 232–255 (24 levels)
            gray_code = 232 + int(v * 23)
            char = "  "
            if use_bg:
                row_str += f"{ESC}48;5;{gray_code}m{char}"
            else:
                row_str += f"{ESC}38;5;{gray_code}m{char}"
        row_str += RESET
        print(row_str)


import torchvision.datasets as D
import torchvision.transforms as T
import numpy as np

# load a single MNIST example as a numpy array
mnist = D.MNIST(root="./data", train=False, download=True, transform=T.ToTensor())
img, label = mnist[0]            # img is a torch.Tensor shape (1,28,28)
arr = img.squeeze(0).numpy()     # now shape (28,28), values in [0,1]

# print("label =", label)
# print_mnist_gray(arr, use_bg=True)

def print_mnist_gray_row(imgs, use_bg=True, separator="  "):
    """
    imgs: list of 2D arrays (all same H×W, values in [0,1])
    use_bg: if True, color background; else color foreground
    separator: string to put between images
    """
    # print(imgs.shape)
    # sdf
    ESC    = "\x1b["
    RESET  = ESC + "0m" # reset terminal color
    n_imgs = len(imgs)
    H, W   = imgs[0].shape
    
    # precompute the per-image, per-pixel ANSI codes into strings
    # so we only convert floats→codes once
    str_imgs = []
    for img in imgs:
        lines = []
        for i in range(H):
            row = []
            for j in range(W):
                v = float(img[i,j])
                v = max(0.0, min(1.0, v))
                gray_code = 232 + int(v * 23)
                cell = f"{ESC}{'48' if use_bg else '38'};5;{gray_code}m  "
                row.append(cell)
            lines.append("".join(row) + RESET)
        # lines[-1] += RESET
        str_imgs.append(lines)
    
    # now print row by row, concatenating across images
    for i in range(H):
        line = separator.join(str_imgs[k][i] for k in range(n_imgs))
        print(line)

# print(mnist[0])
imgs = []
for i in range(3):
    img, label = mnist[i]
    imgs.append(img.squeeze(0).numpy())
# imgs = mnist[0]
# # print(imgs.shape)
# imgs = [x[0] for x in imgs]
import torch
print_mnist_gray_row(imgs)


'''
def print_mnist_ascii(img, use_bg=True):
    """
    img: 2D numpy array or torch.Tensor with values in [0,1]
    use_bg: if True, color the background; else color the foreground text
    """
    ESC = "\x1b["
    RESET = ESC + "0m"
    H, W = img.shape

    for i in range(H):
        row_str = ""
        for j in range(W):
            v = float(img[i, j])
            # clamp just in case
            v = max(0.0, min(1.0, v))
            # map to 0–255
            color = int(v * 255)
            # choose a “block” character
            char = "  "  # two spaces look like a square cell
            # or use "█" for a tighter cell
            if use_bg:
                row_str += f"{ESC}48;5;{color}m{char}"
            else:
                row_str += f"{ESC}38;5;{color}m{char}"
        row_str += RESET  # reset at end of each line
        print(row_str)

def print_mnist_gray(img, use_bg=True):
    ESC = "\x1b["
    RESET = ESC + "0m"
    H, W = img.shape

    for i in range(H):
        row_str = ""
        for j in range(W):
            v = float(img[i, j])
            v = max(0.0, min(1.0, v))
            # map v∈[0,1] to codes 232–255 (24 levels)
            gray_code = 232 + int(v * 23)
            char = "  "
            if use_bg:
                row_str += f"{ESC}48;5;{gray_code}m{char}"
            else:
                row_str += f"{ESC}38;5;{gray_code}m{char}"
        row_str += RESET
        print(row_str)


import torchvision.datasets as D
import torchvision.transforms as T
import numpy as np

# load a single MNIST example as a numpy array
mnist = D.MNIST(root="./data", train=False, download=True, transform=T.ToTensor())
img, label = mnist[0]            # img is a torch.Tensor shape (1,28,28)
arr = img.squeeze(0).numpy()     # now shape (28,28), values in [0,1]

# print("label =", label)
# print_mnist_gray(arr, use_bg=True)

def print_mnist_gray_row(imgs, use_bg=True, separator="  "):
    """
    imgs: list of 2D arrays (all same H×W, values in [0,1])
    use_bg: if True, color background; else color foreground
    separator: string to put between images
    """
    # print(imgs.shape)
    # sdf
    ESC    = "\x1b["
    RESET  = ESC + "0m" # reset terminal color
    n_imgs = len(imgs)
    H, W   = imgs[0].shape
    
    # precompute the per-image, per-pixel ANSI codes into strings
    # so we only convert floats→codes once
    str_imgs = []
    for img in imgs:
        lines = []
        for i in range(H):
            row = []
            for j in range(W):
                v = float(img[i,j])
                v = max(0.0, min(1.0, v))
                gray_code = 232 + int(v * 23)
                cell = f"{ESC}{'48' if use_bg else '38'};5;{gray_code}m  "
                row.append(cell)
            lines.append("".join(row) + RESET)
        # lines[-1] += RESET
        str_imgs.append(lines)
    
    # now print row by row, concatenating across images
    for i in range(H):
        line = separator.join(str_imgs[k][i] for k in range(n_imgs))
        print(line)

# print(mnist[0])
imgs = []
for i in range(3):
    img, label = mnist[i]
    imgs.append(img.squeeze(0).numpy())
# imgs = mnist[0]
# # print(imgs.shape)
# imgs = [x[0] for x in imgs]
import torch
print_mnist_gray_row(imgs)
'''