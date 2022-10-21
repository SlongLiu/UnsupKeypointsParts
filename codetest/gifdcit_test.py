import os, sys
import os.path as osp
sys.path.append(osp.dirname(sys.path[0]))

from utils.utils import gifdict

def main():
    x = {
        'a': 1,
        'b': 2,
        'c': 3
    }
    xx = {
        'x': x,
        'y': 4,
        'z': 5
    }

    print(gifdict(xx, 'x.c', 'dasdsa'))


if __name__ == "__main__":
    main()