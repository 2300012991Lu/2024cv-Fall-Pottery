## Here you may implement the evaluation method and may call some necessary modules from utils.model_utils.py
## Derive the test function by yourself and implement proper metric such as Dice similarity coeffcient (DSC)[4];
# Jaccard distance[5] and Mean squared error (MSE), etc. following the handout in model_utilss.py

import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset, FragmentDataLoader, available_device
from utils.model import Generator, Discriminator
from utils.model_utils import posprocessing
from utils.visualize import plot
from scipy import ndimage

def test(args):
    # TODO
    # You can also implement this function in training procedure, but be sure to
    # evaluate the model on test set and reserve the option to save both quantitative
    # and qualitative (generated .vox or visualizations) images.   

    if args.check_data_only:

        output = np.load(args.out+f'/gen_{args.check_index}.npy').squeeze()
        # dataset = FragmentDataset('./data_voxelized/data', train=True, transform = lambda x : ndimage.zoom(x, zoom=(0.5,0.5,0.5)))
        # output = torch.from_numpy(output)

        # print(error(output, output))
        # return

        for i in np.random.choice(len(output), (5,)).tolist():
            arr = output[i]
            print(arr.shape)
            plot(posprocessing(arr, None), None)

        return
    
    def count_similar(real_vox, fake_vox, threshold=0.01):
        real_vox = torch.flatten(real_vox, start_dim=1)
        fake_vox = torch.flatten(fake_vox, start_dim=1)
        vox_shape = real_vox.shape
        real_vox = torch.where(real_vox > 0.5, 1, 0)
        fake_vox = torch.where(fake_vox > 0.5, 1, 0)
        err = torch.count_nonzero((fake_vox - real_vox).long(), dim=1)
        correct = torch.where(err < threshold * vox_shape[1], 1, 0)
        acc = correct.count_nonzero().cpu().item() / vox_shape[0]
        return acc
    
    dir_dataset = args.dataset
    dir_checkpoint = args.checkpoint
    dir_output = args.out

    n_epoch = args.epoch
    batch_size = args.batch_size

    dataset = FragmentDataset(dir_dataset, train=False, transform = lambda x : ndimage.zoom(x, zoom=(0.5,0.5,0.5)))
    dataloader = FragmentDataLoader(dataset, shuffle=False, batch_size=64)

    n_batch = len(dataloader)

    G = Generator(cube_len=32)
    G.load_state_dict(torch.load(dir_checkpoint+'/model_last.pth', weights_only=True))
    G = G.to(available_device)

    print(f'Use device : {available_device}')
    print('-'*10, 'start testing', '-'*10)

    # for epoch in range(n_epoch):

    total = 0
    total_correct = 0
    print(len(dataset))

    with torch.no_grad():
        for epoch in range(1):

            for i, (voxels, vox_frag, _) in enumerate(dataloader):

                size = len(voxels)

                voxels = voxels.to(available_device)
                vox_frag = vox_frag.to(available_device)

                fake_vox = G(vox_frag)

                correct = int(count_similar(voxels, fake_vox, threshold=0.08) * size)
                total_correct += correct
                total += size

                print('[%3d] Batch acc = %.6f' % (i, correct / size))

        print('Test acc = %.6f' % (total_correct / total))


    return