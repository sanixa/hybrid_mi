import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import torch
from torch.autograd import Variable
import random

import model, log

### import tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools/lpips_pytorch'))
from utils import *
import lpips_pytorch as ps
from LBFGS_pytorch import FullBatchLBFGS

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST

### Hyperparameters
LAMBDA2 = 0.02
LAMBDA3 = 0.01
LBFGS_LR = 0.015

#############################################################################################################
# main optimization function
# ############################################################################################################
class LatentZ(torch.nn.Module):
    def __init__(self, init_val):
        super(LatentZ, self).__init__()
        self.z = torch.nn.Parameter(init_val.data)

    def forward(self):
        return self.z

    def reinit(self, init_val):
        self.z = torch.nn.Parameter(init_val.data)


class Loss(torch.nn.Module):
    def __init__(self, args, netG, netC, distance, if_norm_reg=False, z_dim=100):
        super(Loss, self).__init__()
        self.netG = netG
        self.netC = netC
        self.args = args
        self.lpips_model = ps.PerceptualLoss()

        ### loss
        if distance == 'l2':
            print('Use distance: l2')
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
            self.loss_lpips_fn = lambda x, y: 0.

        elif distance == 'l2-lpips':
            print('Use distance: lpips + l2')
            self.loss_lpips_fn = lambda x, y: self.lpips_model.forward(x, y, normalize=False).view(-1)
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
        '''
        self.distance = distance
        self.lpips_model = ps.PerceptualLoss()
        self.netG = netG
        self.if_norm_reg = if_norm_reg
        self.z_dim = z_dim
        self.label = LongTensor(np.tile(np.arange(10), 2)).cuda()

        ### loss
        if distance == 'l2':
            print('Use distance: l2')
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
            self.loss_lpips_fn = lambda x, y: 0.

        elif distance == 'l2-lpips':
            print('Use distance: lpips + l2')
            self.loss_lpips_fn = lambda x, y: self.lpips_model.forward(x, y, normalize=False).view(-1)
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
        '''
    def forward(self, z, x_gt_ch3, y_gt):
        '''
        self.x_hat = self.netG(z, self.label)
        self.x_hat = torch.cat((self.x_hat, self.x_hat, self.x_hat), 1)
        self.x_hat = torch.clamp(self.x_hat, min=0.0, max=1.0)
        x_gt = torch.unsqueeze(x_gt, 1)
        x_gt = torch.cat((x_gt, x_gt, x_gt), 1)
        x_gt = torch.clamp(x_gt, min=0.0, max=1.0)
        # print(f'{torch.max(self.x_hat)} {torch.min(self.x_hat)} {torch.median(self.x_hat)}')
        # print(f'{torch.max(x_gt)} {torch.min(x_gt)} {torch.median(x_gt)}')
        self.loss_lpips = self.loss_lpips_fn(self.x_hat, x_gt)
        self.loss_l2 = self.loss_l2_fn(self.x_hat, x_gt)
        self.vec_loss = LAMBDA2 * self.loss_lpips + self.loss_l2

        if self.if_norm_reg:
            z_ = z.view(-1, self.z_dim)
            norm = torch.sum(z_ ** 2, dim=1)
            norm_penalty = (norm - self.z_dim) ** 2
            self.vec_loss += LAMBDA3 * norm_penalty

        return self.vec_loss
        '''

        batchsize = x_gt_ch3.size()[0]
        #self.label = torch.cuda.LongTensor(y_gt).cuda()
        self.label = torch.cuda.LongTensor(np.tile(np.arange(10), 2)).cuda()
        self.gen_img_ch3 = self.netG(z, self.label).view(batchsize, 3, self.args.resolution, self.args.resolution)
        #gen_img_ch3 = torch.cat((self.gen_img, self.gen_img, self.gen_img), 1)
        self.gen_img_ch3 = torch.clamp(self.gen_img_ch3, min=0.0, max=1.0)

        #x_gt_ch3 = torch.cat((x_gt, x_gt, x_gt), 1)
        x_gt_ch3 = torch.clamp(x_gt_ch3, min=0.0, max=1.0)

        self.gen_img_ch3 = self.gen_img_ch3.float()
        x_gt_ch3 = x_gt_ch3.float()
        self.loss_lpips = self.loss_lpips_fn(self.gen_img_ch3, x_gt_ch3)
        self.loss_l2 = self.loss_l2_fn(self.gen_img_ch3, x_gt_ch3)

        loss_l2_confident_vector = torch.mean((self.netC(self.gen_img_ch3) - self.netC(x_gt_ch3)) ** 2, dim=[1])

        self.vec_loss = loss_l2_confident_vector# + LAMBDA2 * self.loss_lpips + LAMBDA3 * self.loss_l2

        return self.vec_loss


def optimize_z_lbfgs(args,
                     loss_model,
                     init_val,
                     query_imgs,
                     query_target,
                     save_dir,
                     max_func):
    ### store results
    all_loss = []
    all_z = []
    all_x_hat = []

    BATCH_SIZE = args.batch_size
    ### run the optimization for all query data
    size = len(query_imgs)
    for i in tqdm(range(size // BATCH_SIZE)):
        save_dir_batch = os.path.join(save_dir, str(i))

        try:
            x_batch = query_imgs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            y_batch = query_target[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            x_gt = torch.from_numpy(x_batch).cuda()

            if os.path.exists(save_dir_batch) and False:
                pass
            else:
                visualize_gt(x_batch, check_folder(save_dir_batch))

                ### initialize z
                z = Variable(torch.FloatTensor(init_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])).cuda()
                z_model = LatentZ(z)

                ### LBFGS optimizer
                optimizer = FullBatchLBFGS(z_model.parameters(), lr=LBFGS_LR, history_size=20, line_search='Wolfe',
                                           debug=False)

                ### optimize
                loss_progress = []

                def closure():
                    optimizer.zero_grad()
                    vec_loss = loss_model.forward(z_model.forward(), x_gt, y_batch)
                    vec_loss_np = vec_loss.detach().cpu().numpy()
                    loss_progress.append(vec_loss_np)
                    final_loss = torch.mean(vec_loss)
                    return final_loss

                for step in range(max_func):
                    loss_model.forward(z_model.forward(), x_gt, y_batch)
                    final_loss = closure()
                    final_loss.backward()

                    options = {'closure': closure, 'current_loss': final_loss, 'max_ls': 20}
                    obj, grad, lr, _, _, _, _, _ = optimizer.step(options)

                    if step % (max_func//10) == 0:
                        ### store init
                        x_hat_curr = loss_model.gen_img_ch3.data.cpu().numpy()
                        x_hat_curr = np.transpose(x_hat_curr, [0, 2, 3, 1])
                        vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()
                        visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize init

                    if step == max_func - 1:
                        vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()
                        z_curr = z_model.z.data.cpu().numpy()
                        x_hat_curr = loss_model.gen_img_ch3.data.cpu().numpy()
                        x_hat_curr = np.transpose(x_hat_curr, [0, 2, 3, 1])

                        loss_lpips = loss_model.loss_lpips.data.cpu().numpy()
                        loss_l2 = loss_model.loss_l2.data.cpu().numpy()
                        save_files(save_dir_batch, ['l2', 'lpips'], [loss_l2, loss_lpips])

                        ### store results
                        visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize finale
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, required=True,
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--resolution', type=int, default=32,
                        help='target image resolution')
    parser.add_argument('--seed', '-seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--data_num', '-dnum', type=int, default=100,
                        help='the number of query images to be considered')
    parser.add_argument('--batch_size', '-bs', type=int, default=20,
                        help='batch size (should not be too large for better optimization performance)')
    parser.add_argument('--distance', '-dist', type=str, default='l2-lpips', choices=['l2', 'l2-lpips'],
                        help='the objective function type')
    parser.add_argument('--gan-path', '-gdir', type=str, default=None,
                        help='file path of generator')
    parser.add_argument('--cnn-path', '-cdir', type=str, default=None,
                        help='file path of classifier')
    parser.add_argument('--maxfunc', '-mf', type=int, default=1000,
                        help='the maximum number of function calls (for scipy optimizer)')
    parser.add_argument('--pos_data_dir', '-posdir', type=str,
                        help='the directory for the positive (training) query images set')
    parser.add_argument('--neg_data_dir', '-negdir', type=str,
                        help='the directory for the negative (testing) query images set')
    parser.add_argument('--dataset_name', type=str, default='cifar', choices=['mnist', 'cifar'],
                        help='dataset name')
    parser.add_argument('--LAMBDA2', '-l2', type=float, default=None, 
                        help='LAMBDA2 value for tuning')
    parser.add_argument('--LAMBDA3', '-l3', type=float, default=None, 
                        help='LAMBDA3 value for tuning')
    parser.add_argument('--LBFGS_LR', '-lr', type=float, default=None, 
                        help='LBFGS_LR value for tuning')
    args = parser.parse_args()
    Z_DIM = 100

    global LAMBDA2
    if args.LAMBDA2 != None:
        LAMBDA2 = args.LAMBDA2
    global LAMBDA3
    if args.LAMBDA3 != None:
        LAMBDA3 = args.LAMBDA3
    global LBFGS_LR
    if args.LBFGS_LR != None:
        LBFGS_LR = args.LBFGS_LR

    ### save dir
    save_dir = os.path.join('result', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = log.get_logger(os.path.join(save_dir, 'exp.log'))
    logger.info('LAMBDA2, LAMBDA3, LBFGS_LR: {}, {}, {}'.format(LAMBDA2, LAMBDA3, LBFGS_LR))
    logger.info('args: {}'.format(args))

    ### set up Generator
    if not os.path.isfile(args.gan_path):
        sys.exit("model does not exist")
    netG = model.GeneratorDCGAN_cifar(z_dim=100, model_dim=64, num_classes=10).cuda()
    netG.load_state_dict(torch.load(args.gan_path))
    netG.eval()

    ### set up classifier
    if not os.path.isfile(args.cnn_path):
        sys.exit("model does not exist")
    netC = model.ResNet18v2_cifar10().cuda()
    netC.load_state_dict(torch.load(args.cnn_path))
    netC.eval()

    if args.seed == None:
        seed = np.random.randint(1000, size=1)[0]
    else:
        seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info('random seed: {}'.format(seed))

    ### define loss
    loss_model = Loss(args, netG, netC, args.distance, if_norm_reg=False, z_dim=Z_DIM)

    ### initialization
    init_val_np = np.random.normal(size=(Z_DIM))
    init_val_np = init_val_np / np.sqrt(np.mean(np.square(init_val_np)) + 1e-8)
    init_val = np.tile(init_val_np, (args.data_num, 1)).astype(np.float32)
    init_val_pos = init_val
    init_val_neg = init_val

    if args.dataset_name == 'mnist':
        transform = transforms.ToTensor()
        train_set = MNIST(root="./data/MNIST", download=True, train=True, transform=transform)
        test_set = MNIST(root="./data/MNIST", download=True, train=False, transform=transform)
    elif args.dataset_name == 'cifar':
        transform = transforms.Compose([
                        #transforms.Grayscale(1),
                        transforms.ToTensor()])

        train_set = CIFAR10(root="./data/CIFAR10", download=True, train=True, transform=transform)
        test_set = CIFAR10(root="./data/CIFAR10", download=True, train=False, transform=transform)


    #indices = np.loadtxt('index_500_dcgan.txt', dtype=np.int_)
    #train_set = torch.utils.data.Subset(train_set, indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    train_data, train_target = next(iter(train_loader))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)
    test_data, test_target = next(iter(test_loader))

    ### positive ###
    #pos_data_paths = get_filepaths_from_dir(args.pos_data_dir, ext='png')[: args.data_num]
    #pos_query_imgs = np.array([read_image(f, resolution=args.resolution) for f in pos_data_paths])
    pos_query_imgs = train_data.cpu().detach().numpy()
    pos_query_target = train_target.cpu().detach().numpy()

    query_loss, query_z, query_xhat = optimize_z_lbfgs(args,
                                                       loss_model,
                                                       init_val_pos,
                                                       pos_query_imgs,
                                                       pos_query_target,
                                                       check_folder(os.path.join(save_dir, 'pos_results')),
                                                       args.maxfunc)
    save_files(save_dir, ['pos_loss'], [query_loss])
    
    ### negative ###
    #neg_data_paths = get_filepaths_from_dir(args.neg_data_dir, ext='png')[: args.data_num]
    #neg_query_imgs = np.array([read_image(f, resolution=args.resolution) for f in neg_data_paths])
    neg_query_imgs = test_data.cpu().detach().numpy()
    neg_query_target = test_target.cpu().detach().numpy()

    query_loss, query_z, query_xhat = optimize_z_lbfgs(args,
                                                       loss_model,
                                                       init_val_neg,
                                                       neg_query_imgs,
                                                       neg_query_target,
                                                       check_folder(os.path.join(save_dir, 'neg_results')),
                                                       args.maxfunc)
    save_files(save_dir, ['neg_loss'], [query_loss])

if __name__ == '__main__':
    main()
