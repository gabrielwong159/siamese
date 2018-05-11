from os import listdir
from pathlib import Path

"""
{
    "Alphabet_of_the_Magi": {
        "character01": ['a.png', 'b.png', ...],
        "character02": [...],
        ...
    },
    "Anglo-Saxon_Futhorc": {
        "character01": [...],
        ...    
    },
    ...
}
"""


def get_files(dataset='train'):
    if dataset == 'train':
        src = Path('data', 'images_background')
    elif dataset == 'test':
        src = Path('data', 'images_evaluation')
    else:
        raise ValueError('Invalid dataset parameter provided')

    out = {}
    for alphabet in listdir(src):
        chars = {}
        for character in listdir(src / alphabet):
            folder = src / alphabet / character
            chars[character] = [str(folder / f) for f in listdir(folder)]
        out[alphabet] = chars

    return out


if __name__ == '__main__':
    d = get_files()
    d = get_files('test')
