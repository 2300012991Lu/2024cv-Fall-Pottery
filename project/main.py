from training import train
from test import test



def main():

    # your hyperparam list MAY INCLUDE:
    # 1. Z_latent_space
    # 2. G_lr
    # 3. D_lr  (learning rate for Discriminator)
    # 4. betas if you are going to use Adam optimizer
    # 5. Resolution for input data
    # 6. Training Epochs
    # 7. Test per epoch
    # 8. Batch Size
    # 9. Dataset Dir
    # 10. Load / Save model Device
    # 11. test result save dir
    # 12. device!
    # .... (maybe there exists more hyperparams to be appointed)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--G_lr', type=float, default=0.0002)
    parser.add_argument('--D_lr', type=float, default=0.0002)
    parser.add_argument('--D_importance', type=float, default=0.1)

    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--dataset', type=str, default='./data_voxelized/data')
    parser.add_argument('--out', type=str, default='./out')

    parser.add_argument('--check_data_only', action='store_true')
    parser.add_argument('--check_index', type=int, default=1)

    args = parser.parse_args()


    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()

