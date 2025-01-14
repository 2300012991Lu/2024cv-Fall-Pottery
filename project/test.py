## Here you may implement the evaluation method and may call some necessary modules from utils.model_utils.py
## Derive the test function by yourself and implement proper metric such as Dice similarity coeffcient (DSC)[4];
# Jaccard distance[5] and Mean squared error (MSE), etc. following the handout in model_utilss.py

import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch import nn
from utils.FragmentDataset import FragmentDataset, FragmentDataLoader, available_device
from utils.model import Generator, Discriminator, JaccardDistance
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
    
    def cal_similarity(real_vox, fake_vox, threshold=0.2):
        flat_real = nn.Flatten()(real_vox).unsqueeze(2)
        flat_fake = nn.Flatten()(fake_vox).unsqueeze(2)
        real_vox = torch.where(real_vox > 0.5, 1, 0)
        fake_vox = torch.where(fake_vox > 0.5, 1, 0)
        compare = torch.cat([flat_real, flat_fake], dim=2)
        cap = torch.min(compare, dim=2)[0].sum(1)
        cup = torch.max(compare, dim=2)[0].sum(1)
        correct = torch.where((cup - cap) / cup < threshold, 1, 0)
        acc = correct.count_nonzero().cpu().item() / len(real_vox)
        return acc
    
    dir_dataset = args.dataset
    dir_checkpoint = args.checkpoint
    dir_output = args.out

    n_epoch = args.epoch
    batch_size = args.batch_size

    dataset = FragmentDataset(dir_dataset, train=False, transform = lambda x : ndimage.zoom(x, zoom=(0.5,0.5,0.5)))
    dataloader = FragmentDataLoader(dataset, shuffle=False, batch_size=batch_size)

    n_batch = len(dataloader)

    G = Generator(cube_len=32)
    D = Discriminator(resolution=32)
    checkpoint = torch.load(dir_checkpoint+'/model_5.pth', weights_only=False)
    G.load_state_dict(checkpoint['Generator'])
    D.load_state_dict(checkpoint['Discriminator'])
    G = G.to(available_device)
    D = D.to(available_device)

    def cal_authenticity(fake_target):
        fake_target = fake_target.to(available_device)
        correct = torch.where(fake_target == 1, 1, 0)
        acc = correct.count_nonzero().cpu().item() / len(fake_target)
        return acc


    print(f'Use device : {available_device}')
    print('-'*10, 'start testing', '-'*10)

    # for epoch in range(n_epoch):

    total = 0
    total_similarity = 0
    total_authenticity = 0
    print(len(dataset))

    with torch.no_grad():
        for epoch in range(1):

            for i, (voxels, vox_frag, _) in enumerate(dataloader):

                size = len(voxels)

                voxels = voxels.to(available_device)
                vox_frag = vox_frag.to(available_device)

                fake_vox = G(vox_frag)

                # for i in np.random.choice(len(fake_vox), (5,)).tolist():
                #     arr = fake_vox[i].cpu().numpy().squeeze()
                #     print(arr.shape)
                #     plot(posprocessing(arr, None), f'./out/Voxel_5_{i}.html')

                # exit(0)
                fake_target = D(fake_vox)

                similarity = cal_similarity(voxels, fake_vox, threshold=0.4)
                authenticity = cal_authenticity(fake_target)
                total += size
                total_similarity += int(similarity * size)
                total_authenticity += int(authenticity * size)

                print('[%3d] Batch [Similarity = %.6f] [Authenticity = %.6f]' % (i, similarity, authenticity))

        print('Test [Similarity(By Label) = %.6f] [Authenticity(By Discriminator) = %.6f]' % (total_similarity / total, total_authenticity / total))


    return