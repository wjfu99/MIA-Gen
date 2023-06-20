import numpy as np
import os
import sys
import pickle
import argparse
from tqdm import tqdm

### import tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools/lpips_pytorch'))
import lpips_pytorch as ps
from scipy.optimize import minimize

### import victim models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../gan_models/vaegan'))

### Hyperparameters
LAMBDA2 = 0.2
LAMBDA3 = 0.001
RANDOM_SEED = 1000

import os
import numpy as np
import fnmatch
import PIL.Image
import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

NCOLS = 5


def check_folder(dir):
    '''
    create a new directory if it doesn't exist
    :param dir:
    :return:
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def save_files(save_dir, file_name_list, array_list):
    '''
    save a list of array with the given name
    :param save_dir: the directory for saving the files
    :param file_name_list: the list of the file names
    :param array_list: the list of arrays to be saved
    '''
    assert len(file_name_list) == len(array_list)

    for i in range(len(file_name_list)):
        np.save(os.path.join(save_dir, file_name_list[i]), array_list[i], allow_pickle=False)

def get_filepaths_from_dir(data_dir, ext):
    '''
    return all the file paths with extension 'ext' in the given directory 'data_dir'
    :param data_dir: the data directory
    :param ext: the extension type
    :return:
        path_list: list of file paths
    '''
    pattern = '*.' + ext
    path_list = []
    for d, s, fList in os.walk(data_dir):
        for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
                path_list.append(os.path.join(d, filename))
    return sorted(path_list)


def read_image(filepath, resolution=64, cx=89, cy=121):
    '''
    read,crop and scale an image given the path
    :param filepath:  the path of the image file
    :param resolution: desired size of the output image
    :param cx: x_coordinate of the crop center
    :param cy: y_coordinate of the crop center
    :return:
        image in range [-1,1] with shape (resolution,resolution,3)
    '''

    img = np.asarray(PIL.Image.open(filepath))
    shape = img.shape

    if shape == (resolution, resolution, 3):
        pass
    else:
        img = img[cy - 64: cy + 64, cx - 64: cx + 64]
        resize_factor = 128 // resolution
        img = img.astype(np.float32)
        while resize_factor > 1:
            img = (img[0::2, 0::2, :] + img[0::2, 1::2, :] + img[1::2, 0::2, :] + img[1::2, 1::2, :]) * 0.25
            resize_factor -= 1
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    img = img.astype(np.float32) / 255.
    img = img * 2 - 1.
    return img


####################################################
## visualize
####################################################
def inverse_transform(imgs):
    '''
    normalize the image to be of range [0,1]
    :param imgs: input images
    :return:
        images with value range [0,1]
    '''
    imgs = (imgs + 1.) / 2.
    return imgs


def visualize_gt(imgs, save_dir):
    '''
    visualize the ground truth images and save
    :param imgs: input images with value range [-1,1]
    :param save_dir: directory for saving the results
    '''
    plt.figure(1)
    num_imgs = len(imgs)
    imgs = np.clip(inverse_transform(imgs), 0., 1.)
    NROWS = int(np.ceil(float(num_imgs) / float(NCOLS)))
    for i in range(num_imgs):
        plt.subplot(NROWS, NCOLS, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'input.png'))
    plt.close()


def visualize_progress(imgs, loss, save_dir, counter):
    '''
    visualize the optimization results and save
    :param imgs: input images with value range [-1,1]
    :param loss: the corresponding loss values
    :param save_dir: directory for saving the results
    :param counter: number of the function evaluation
    :return:
    '''
    plt.figure(2)
    num_imgs = len(imgs)
    imgs = np.clip(inverse_transform(imgs), 0., 1.)
    NROWS = int(np.ceil(float(num_imgs) / float(NCOLS)))
    for i in range(num_imgs):
        plt.subplot(NROWS, NCOLS, i + 1)
        plt.imshow(imgs[i])
        plt.title('loss: %.4f' % loss[i], fontdict={'fontsize': 8, 'color': 'blue'})
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'output_%d.png' % counter))
    plt.close()


def visualize_samples(img_r01, save_dir):
    plt.figure(figsize=(20, 20))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(img_r01[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'samples.png'))


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, required=True,
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--gan_model_dir', '-gdir', type=str, required=True,
                        help='directory for the Victim GAN model')
    parser.add_argument('--pos_data_dir', '-posdir', type=str,
                        help='the directory for the positive (training) query images set')
    parser.add_argument('--neg_data_dir', '-negdir', type=str,
                        help='the directory for the negative (testing) query images set')
    parser.add_argument('--data_num', '-dnum', type=int, default=5,
                        help='the number of query images to be considered')
    parser.add_argument('--batch_size', '-bs', type=int, default=1,
                        help='batch size (should not be too large for better optimization performance)')
    parser.add_argument('--resolution', '-resolution', type=int, default=64,
                        help='generated image resolution')
    parser.add_argument('--initialize_type', '-init', type=str, default='random',
                        choices=['zero',  # 'zero': initialize the z to be zeros
                                 'random',  # 'random': use normal distributed initialization
                                 'nn',  # 'nn': use the result of the knn as the initialization
                                 ],
                        help='the initialization techniques')
    parser.add_argument('--nn_dir', '-ndir', type=str,
                        help='the directory for storing the fbb(KNN) results')
    parser.add_argument('--distance', '-dist', type=str, default='l2-lpips', choices=['l2', 'l2-lpips'],
                        help='the objective function type')
    parser.add_argument('--if_norm_reg', '-reg', action='store_true', default=True,
                        help='enable the norm regularizer')
    parser.add_argument('--maxfunc', '-mf', type=int, default=10,
                        help='the maximum number of function calls (for scipy optimizer)')
    return parser.parse_args()


def check_args(args):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## load dir
    assert os.path.exists(args.gan_model_dir)

    ## set up save_dir
    save_dir = os.path.join(os.path.dirname(__file__), 'results/pbb', args.exp_name)
    check_folder(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)

    return args, save_dir, args.gan_model_dir


#############################################################################################################
# main optimization function
#############################################################################################################
class LatentZ(torch.nn.Module):
    def __init__(self, init_val):
        super(LatentZ, self).__init__()
        self.z = torch.nn.Parameter(init_val.data)

    def forward(self):
        return self.z

    def reinit(self, init_val):
        self.z = torch.nn.Parameter(init_val.data)


class Loss(torch.nn.Module):
    def __init__(self, netG, distance, if_norm_reg=False, z_dim=100):
        super(Loss, self).__init__()
        self.distance = distance
        self.lpips_model = ps.PerceptualLoss()
        self.netG = netG
        self.if_norm_reg = if_norm_reg
        self.z_dim = z_dim

        ### loss
        if distance == 'l2':
            print('Use distance: l2')
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
            self.loss_lpips_fn = lambda x, y: 0.

        elif distance == 'l2-lpips':
            print('Use distance: lpips + l2')
            self.loss_lpips_fn = lambda x, y: self.lpips_model.forward(x, y, normalize=False).view(-1)
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])

    def forward(self, z, x_gt):
        self.x_hat = self.netG(z)
        self.loss_lpips = self.loss_lpips_fn(self.x_hat, x_gt)
        self.loss_l2 = self.loss_l2_fn(self.x_hat, x_gt)
        self.vec_loss = LAMBDA2 * self.loss_lpips + self.loss_l2

        if self.if_norm_reg:
            z_ = z.view(-1, self.z_dim)
            norm = torch.sum(z_ ** 2, dim=1)
            norm_penalty = (norm - z_dim) ** 2
            self.vec_loss += LAMBDA3 * norm_penalty

        return self.vec_loss


def optimize_z_bb(loss_model,
                  init_val,
                  query_imgs,
                  save_dir,
                  max_func):
    ### store results
    all_loss = []
    all_z = []
    all_x_hat = []

    ### run the optimization for all query data
    size = len(query_imgs)
    for i in tqdm(range(size // BATCH_SIZE)):
    # for i in tqdm(range(size // (2*BATCH_SIZE),size // BATCH_SIZE)):
        save_dir_batch = os.path.join(save_dir, str(i))

        try:
            x_batch = query_imgs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            x_gt = torch.from_numpy(x_batch).permute(0, 3, 1, 2).cuda()

            if os.path.exists(save_dir_batch):
                pass
            else:
                visualize_gt(x_batch, check_folder(save_dir_batch))

                ### optimize
                loss_progress = []

                def objective(z):
                    z_ = torch.from_numpy(z).float().view(1, -1, 1, 1).cuda()
                    vec_loss = loss_model.forward(z_, x_gt)
                    vec_loss_np = vec_loss.detach().cpu().numpy()
                    loss_progress.append(vec_loss_np)
                    final_loss = torch.mean(vec_loss)
                    return final_loss.detach().cpu().numpy()

                z0 = init_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                options = {'maxiter': max_func,
                           'disp': 1}
                res = minimize(objective, z0, method='Powell', options=options)
                z_curr = res.x
                vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()
                x_hat_curr = loss_model.x_hat.data.cpu().numpy()
                x_hat_curr = np.transpose(x_hat_curr, [0, 2, 3, 1])

                loss_lpips = loss_model.loss_lpips.data.cpu().numpy()
                loss_l2 = loss_model.loss_l2.data.cpu().numpy()
                save_files(save_dir_batch, ['l2', 'lpips'], [loss_l2, loss_lpips])

                ### store results
                visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, len(loss_progress))  # visualize finale
                all_loss.append(vec_loss_curr)
                all_z.append(z_curr)
                all_x_hat.append(x_hat_curr)

                save_files(save_dir_batch,
                           ['full_loss', 'z', 'xhat', 'loss_progress'],
                           [vec_loss_curr, z_curr, x_hat_curr, np.array(loss_progress)])

        except KeyboardInterrupt:
            print('Stop optimization\n')
            break
    try:
        all_loss = np.concatenate(all_loss)
        all_z = np.concatenate(all_z)
        all_x_hat = np.concatenate(all_x_hat)
    except:
        all_loss = np.array(all_loss)
        all_z = np.array(all_z)
        all_x_hat = np.array(all_x_hat)
    return all_loss, all_z, all_x_hat


#############################################################################################################
# main
#############################################################################################################
def main():
    args, save_dir, load_dir = check_args(parse_arguments())

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    ### set up Generator
    network_path = os.path.join(load_dir, 'netG.pt')
    netG = torch.load(network_path).cuda()
    netG.eval()
    Z_DIM = netG.deconv1.module.in_channels
    resolution = args.resolution

    ### define loss
    loss_model = Loss(netG, args.distance, if_norm_reg=False, z_dim=Z_DIM)

    ### initialization
    if args.initialize_type == 'zero':
        init_val = np.zeros((args.data_num, Z_DIM, 1, 1))
        init_val_pos = init_val
        init_val_neg = init_val

    elif args.initialize_type == 'random':
        np.random.seed(RANDOM_SEED)
        init_val_np = np.random.normal(size=(Z_DIM, 1, 1))
        init_val_np = init_val_np / np.sqrt(np.mean(np.square(init_val_np)) + 1e-8)
        init_val = np.tile(init_val_np, (args.data_num, 1, 1, 1)).astype(np.float32)
        init_val_pos = init_val
        init_val_neg = init_val

    elif args.initialize_type == 'nn':
        idx = 0
        init_val_pos = np.load(os.path.join(args.nn_dir, 'pos_z.npy'))[:, idx, :]
        init_val_pos = np.reshape(init_val_pos, [len(init_val_pos), Z_DIM, 1, 1])
        init_val_neg = np.load(os.path.join(args.nn_dir, 'neg_z.npy'))[:, idx, :]
        init_val_neg = np.reshape(init_val_neg, [len(init_val_neg), Z_DIM, 1, 1])
    else:
        raise NotImplementedError

    ### positive ###
    pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext='png')[: args.data_num]
    pos_query_imgs = np.array([read_image(f, resolution) for f in pos_data_paths])
    query_loss, query_z, query_xhat = optimize_z_bb(loss_model,
                                                    init_val_pos,
                                                    pos_query_imgs,
                                                    check_folder(os.path.join(save_dir, 'pos_results')),
                                                    args.maxfunc)
    save_files(save_dir, ['pos_loss'], [query_loss])

    ### negative ###
    neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext='png')[: args.data_num]
    neg_query_imgs = np.array([read_image(f, resolution) for f in neg_data_paths])
    query_loss, query_z, query_xhat = optimize_z_bb(loss_model,
                                                    init_val_neg,
                                                    neg_query_imgs,
                                                    check_folder(os.path.join(save_dir, 'neg_results')),
                                                    args.maxfunc)
    save_files(save_dir, ['neg_loss'], [query_loss])


if __name__ == '__main__':
    main()
