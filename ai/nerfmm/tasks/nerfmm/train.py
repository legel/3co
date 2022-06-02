import sys
import os
import argparse
from pathlib import Path
import datetime
import shutil
import logging
import imageio

import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import random
import math
sys.path.append(os.path.join(sys.path[0], '../..'))

# from dataloader.with_colmap import DataLoaderWithCOLMAP
from dataloader.with_smartphone import DataLoaderWithSmartphone

from utils.training_utils import set_randomness, mse2psnr, save_checkpoint
from utils.pos_enc import encode_position
from utils.volume_op import volume_sampling, volume_sampling_ndc, volume_rendering
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.comp_ate import compute_ate
from utils.lie_group_helper import compute_angular_distance
from models.nerf_models import OfficialNerf
from models.intrinsics import LearnFocal
from models.poses import LearnPose
from models.pose_net import PoseNet

import wandb


number_of_epochs = 3000
early_termination_epoch = 3001
eval_image_interval = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=number_of_epochs, type=int)
    parser.add_argument('--eval_img_interval', default=eval_image_interval, type=int, help='eval images every this epoch number')
    parser.add_argument('--eval_cam_interval', default=eval_image_interval, type=int, help='eval camera params every this epoch number')

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, type=eval, choices=[True, False])
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data',
                        help='folder contains various scenes')
    parser.add_argument('--scene_name', type=str, default='pillow_scan')
    parser.add_argument('--use_ndc', default=False, type=eval, choices=[True, False])

    parser.add_argument('--nerf_lr', default=0.002, type=float)
    parser.add_argument('--nerf_milestones', default=list(range(0, number_of_epochs, 10)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--nerf_lr_gamma', type=float, default=0.98, help="learning rate milestones gamma")

    parser.add_argument('--learn_focal', default=True, type=eval, choices=[True, False])
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--focal_lr', default=0.002, type=float)
    parser.add_argument('--focal_milestones', default=list(range(0, number_of_epochs, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--focal_lr_gamma', type=float, default=0.94, help="learning rate milestones gamma")

    parser.add_argument('--learn_R', default=True, type=eval, choices=[True, False])
    parser.add_argument('--learn_t', default=True, type=eval, choices=[True, False])
    parser.add_argument('--pose_lr', default=0.001, type=float)
    parser.add_argument('--pose_milestones', default=list(range(0, number_of_epochs, 10)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--pose_lr_gamma', type=float, default=0.98, help="learning rate milestones gamma")

    parser.add_argument('--store_pose_history', type=bool, default=True, help='store pose history to log dir')

    parser.add_argument('--start_refine_pose_epoch', type=int, default=0,
                        help='Set to -1 to init pose from identity matrices. Set to a epoch number >= 0 '
                             'to init poses from COLMAP and start refining them from this epoch.')

    parser.add_argument('--start_refine_focal_epoch', type=int, default=0,
                        help='Set to -1 to init focal from image resolution. Set to a epoch number >= 0 '
                             'to init focals from COLMAP and start refining them from this epoch.')

    parser.add_argument('--start_refine_rgb_epoch', type=int, default=50,
                        help='Set to a epoch number >= 0 to start learning RGB NeRF on top of density NeRF.')

    parser.add_argument('--resize_ratio', type=int, default=1, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=120, help='network hidden unit dimensions')
    parser.add_argument('--train_rand_rows', type=int, default=64, help='rand sample these rows to train')
    parser.add_argument('--train_rand_cols', type=int, default=64, help='rand sample these cols to train')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=5, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train, -1 for all')
    parser.add_argument('--train_skip', type=int, default=60*3, help='skip every this number of imgs')
    parser.add_argument('--eval_img_num', type=int, default=1, help='num of images to eval')
    parser.add_argument('--eval_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--alias', type=str, default='', help="experiments alias")

    parser.add_argument('--depth_loss_initial_importance', default=0.80, type=float)
    parser.add_argument('--depth_loss_final_importance', default=0.01, type=float)


    # parser.add_argument('--depth_loss_exponential_decay_rate', default=1.0, type=float)
    parser.add_argument('--depth_loss_exponential_index', default=2, type=int)
    parser.add_argument('--depth_loss_curve_shape', default=5, type=int)





    parser.add_argument('--rgb_loss_importance', default=10.0, type=float)

    parser.add_argument('--pose_deviation_loss_initial_importance', default=0.0, type=float)
    parser.add_argument('--pose_deviation_loss_exponential_decay_rate', default=3.5, type=float)

    parser.add_argument('--empirical_maximum_depth_loss', default=0.15, type=float)
    parser.add_argument('--empirical_maximum_rgb_loss', default=1.20, type=float)

    parser.add_argument('--depth_variance', default=0.0000001, type=float)

    parser.add_argument('--log_frequency', default=1, type=int)


    parsed_args = parser.parse_args()

    wandb.init(project="nerf--", entity="3co", config=parsed_args)

    return parser.parse_args()


def gen_detail_name(args):
    outstr = 'lr_' + str(args.nerf_lr) + \
             '_gpu' + str(args.gpu_id) + \
             '_seed_' + str(args.rand_seed) + \
             '_resize_' + str(args.resize_ratio) + \
             '_Nsam_' + str(args.num_sample) + \
             '_Ntr_img_'+ str(args.train_img_num) + \
             '_freq_' + str(args.pos_enc_levels) + \
             '_' + str(args.alias) + \
             '_' + str(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    return outstr


# def store_current_pose(pose_net, pose_history_dir, epoch_i):
#     pose_net.eval()

#     num_cams = pose_net.module.num_cams if isinstance(pose_net, torch.nn.DataParallel) else pose_net.num_cams

#     c2w_list = []
#     for i in range(num_cams):
#         c2w = pose_net(i)  # (4, 4)
#         c2w_list.append(c2w)

#     c2w_list = torch.stack(c2w_list)  # (N, 4, 4)
#     c2w_list = c2w_list.detach().cpu().numpy()

#     np.save(os.path.join(pose_history_dir, str(epoch_i).zfill(6)), c2w_list)
#     return


def model_render_image(c2w, rays_cam, t_vals, near, far, H, W, fxfy, model, perturb_t, sigma_noise_std,
                       args, rgb_act_fn):
    """Render an image or pixels.
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an entire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param fxfy:        a float or a (2, ) torch tensor for focal.
    :param perturb_t:   True/False              whether add noise to t.
    :param sigma_noise_std: a float             std dev when adding noise to raw density (sigma).
    :rgb_act_fn:        sigmoid()               apply an activation fn to the raw rgb output to get actual rgb.
    :return:            (someH, someW, 3)       volume rendered images for the input rays.
    """

    # # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    # sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, near, far,
    #                                                                  H, W, fxfy, perturb_t)

    if args.use_ndc:
        sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, near, far,
                                                                         H, W, fxfy, perturb_t)
    else:
        sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling(c2w, rays_cam, t_vals, near, far, perturb_t)



    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)

    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    if args.use_dir_enc:
        ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
        dir_enc = encode_position(ray_dir_world, levels=args.dir_enc_levels, inc_input=args.dir_enc_inc_in)  # (H, W, 27)
        dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, args.num_sample, -1)  # (H, W, N_sample, 27)
    else:
        dir_enc = None

    # inference rgb and density using position and direction encoding.
    rgb_density = model(pos_enc, dir_enc)  # (H, W, N_sample, 4)

    render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn)
    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)
    weight = render_result['weight']

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'sample_pos': sample_pos,  # (H, W, N_sample, 3)
        'depth_map': depth_map,  # (H, W)
        'rgb_density': rgb_density,  # (H, W, N_sample, 4)
        'depth_weights': weight, # (H, W, N_sample),
        't_vals': t_vals_noisy # (N_samples)
    }

    return result


def heatmap_to_pseudo_color(heatmap):
    # https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
    turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]
    turbo_colormap_data_np = np.array(turbo_colormap_data)

    x = heatmap

    # normalize image so that relative depth is clearly hightlighted
    #minimum_depth_value = np.min(x)
    #maximum_depth_value = np.max(x)
    minimum_depth_value = 0.04
    maximum_depth_value = 0.30
    x = (x - minimum_depth_value) / (maximum_depth_value - minimum_depth_value)

    x = x.clip(0, 1)
    a = (x * 255).astype(int)
    b = (a + 1).clip(max=255)
    f = x * 255.0 - a
    pseudo_color = (turbo_colormap_data_np[a] + (turbo_colormap_data_np[b] - turbo_colormap_data_np[a]) * f[..., None])
    pseudo_color[heatmap < 0.0] = 0.0
    pseudo_color[heatmap > 1.0] = 1.0
    return pseudo_color

def eval_one_epoch_img(evaluation_image_pose, scene_train, model, focal_net, pose_param_net,
                       my_devices, args, epoch_i, writer, rgb_act_fn, evaluation_image_index):
    # print("Evaluating image (saving) at epoch {}".format(epoch_i))
    model.eval()
    focal_net.eval()
    pose_param_net.eval()

    fxfy = focal_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(scene_train.H, scene_train.W, fxfy[0], fxfy[1])
    N_img = evaluation_image_pose.shape[0]

    rendered_img_list = []
    rendered_depth_list = []

    for i in range(N_img):
        c2w = evaluation_image_pose[i].to(my_devices)  # (4, 4)

        t_vals = torch.linspace(scene_train.near[evaluation_image_index], scene_train.far[evaluation_image_index], args.num_sample, device=my_devices)  # (N_sample,) sample position

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy,
                                               model, False, 0.0, args, rgb_act_fn)
            rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)
        rendered_color_for_file = (rendered_img.cpu().numpy() * 255).astype(np.uint8)

        # get depth map and convert it to Turbo Color Map
        rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)
        rendered_depth_for_file = rendered_depth.cpu().numpy() 
        rendered_depth_for_file = heatmap_to_pseudo_color(rendered_depth_for_file)
        rendered_depth_for_file = (rendered_depth_for_file * 255).astype(np.uint8)
        # rendered_depth_for_file = np.transpose(rendered_depth_for_file, (1, 2, 0))


        image_out_dir = "{}/{}/hyperparam_experiments".format(scene_train.base_dir, scene_train.scene_name)
        
        N = "{}".format(args.depth_loss_exponential_index)#.replace(".","p")
        k = "{}".format(args.depth_loss_curve_shape)#.replace(".","p")
        # rgb_loss_importance = "{:.3f}".format(args.rgb_loss_importance).replace(".","p")

        experiment_label = "{}_k{}_N{}".format(scene_train.unix_time, k, N)
        
        experiment_dir = Path(os.path.join(image_out_dir, experiment_label))
        experiment_dir.mkdir(parents=True, exist_ok=True)

        color_out_dir = Path("{}/color_nerf_out/".format(experiment_dir))
        color_out_dir.mkdir(parents=True, exist_ok=True)

        color_file_name = os.path.join(color_out_dir, str(i).zfill(4) + '_{}_color.png'.format(epoch_i))
        # print("\nRendering color image of shape {} at {}".format(rendered_color_for_file.shape, color_file_name))
        imageio.imwrite(color_file_name, rendered_color_for_file)

        depth_out_dir = Path("{}/depth_nerf_out/".format(experiment_dir))
        depth_out_dir.mkdir(parents=True, exist_ok=True)

        depth_file_name = os.path.join(depth_out_dir, str(i).zfill(4) + '_{}_depth.png'.format(epoch_i))
        # print("Rendering depth image of shape {} at {}\n".format(rendered_depth_for_file.shape, depth_file_name))
        imageio.imwrite(depth_file_name, rendered_depth_for_file)

        # # for vis
        rendered_img_list.append(rendered_img.cpu().numpy())
        rendered_depth_list.append(rendered_depth.cpu().numpy())

    # random display an eval image to tfboard
    rand_num = np.random.randint(low=0, high=N_img)
    disp_img = np.transpose(rendered_img_list[rand_num], (2, 0, 1))  # (3, H, W)
    disp_depth = rendered_depth_list[rand_num]  # (1, H, W)




    # writer.add_image('eval_img', disp_img, global_step=epoch_i)
    # writer.add_image('eval_depth', disp_depth, global_step=epoch_i)


def eval_one_epoch_traj(scene_train, pose_param_net):
    pose_param_net.eval()

    est_c2ws_train = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])  # (N, 4, 4)
    colmap_c2ws_train = scene_train.c2ws  # (N, 4, 4) torch
    stats_tran, stats_rot, stats_scale = compute_ate(est_c2ws_train, colmap_c2ws_train, align_a2b='sim3')

    return stats_tran, stats_rot, stats_scale


# https://arxiv.org/pdf/2004.05909v1.pdf
def polynomial_decay(current_step, total_steps, start_value, end_value, exponential_index=1, curvature_shape=1):
    return (start_value - end_value) * (1 - current_step**curvature_shape / total_steps**curvature_shape)**exponential_index + end_value


def train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, model, focal_net, pose_param_net,
                    my_devices, args, rgb_act_fn, epoch_i, batch_size=48, reviews_per_batch=1):

    if epoch_i == early_termination_epoch:
        print("Terminating early to speed up hyperparameter search")
        sys.exit(0)

    model.train()

    if epoch_i >= args.start_refine_pose_epoch:
        pose_param_net.train()
    else:
        pose_param_net.eval()

    if epoch_i >= args.start_refine_focal_epoch:
        focal_net.train()
    else:
        focal_net.eval()


    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []

    # # shuffle the training imgs
    # ids = np.arange(N_img)
    # np.random.shuffle(ids)
    
    # get random selection of images from the training data set (stochastic gradient descent)
    if batch_size > N_img:
        batch_size = N_img

    indices_of_selected_random_images = random.sample(range(N_img), batch_size)

    # print("Going through images: {}".format(indices_of_selected_random_images))

    # print("For epoch {} the {} images for training are: {}".format(epoch_i, len(indices_of_selected_random_images), indices_of_selected_random_images))

    # (4) Over time, reduce the reliance on depth coordinates

    # Is it the case that actually COLMAP precision is actually essential?
    weighted_sum_of_losses = torch.tensor(0.0, device=my_devices)
    sum_of_losses = torch.tensor(0.0, device=my_devices)
    sum_of_rgb_losses = torch.tensor(0.0, device=my_devices)
    sum_of_depth_losses = torch.tensor(0.0, device=my_devices)
    sum_of_cam_pos_losses = torch.tensor(0.0, device=my_devices)
    sum_of_cam_rot_losses = torch.tensor(0.0, device=my_devices)
    sum_of_depth_to_rgb_loss_ratios = torch.tensor(0.0, device=my_devices)

    count_of_losses = torch.tensor(0, device=my_devices)
    for review_number in range(reviews_per_batch):
        per_image_rgb_loss = {}
        per_image_depth_loss = {}
        # loss_for_this_review = torch.tensor(0.0, device=my_devices)
        for i in indices_of_selected_random_images:
            # get the randomly selected image
            img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)

            if epoch_i >= args.start_refine_focal_epoch:
                fxfy = focal_net(0)
                ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
            else:
                fxfy = scene_train.focal
                ray_dir_cam = scene_train.ray_dir_cam.to(my_devices)

            if epoch_i >= args.start_refine_pose_epoch:
                c2w = pose_param_net(i)  # (4, 4)
            else:
                with torch.no_grad():
                    c2w = pose_param_net(i)  # (4, 4)

            # sample pixel on an image and their rays for training.
            r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  # (N_select_rows)
            c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  # (N_select_cols)
            ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
            img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)


            t_vals = torch.linspace(scene_train.near[i], scene_train.far[i], args.num_sample, device=my_devices)  # (N_sample,) sample position

            # render an image using selected rays, pose, sample intervals, and the network
            render_result = model_render_image(c2w, ray_selected_cam, t_vals, scene_train.near[i], scene_train.far[i],
                                               scene_train.H, scene_train.W, fxfy,
                                               model, True, 0.0, args, rgb_act_fn)  # (N_select_rows, N_select_cols, 3)
            rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)
            t_vals = render_result['t_vals'] # (N_samples) - updates t_vals (distances along ray to be sampled) after adding noise...


            # implement DS-NeRF here: add a loss from the difference between the depth map from the sensor and the depth map of NeRF
            #nerf_depth = render_result['depth_map'] # (N_select_rows, N_select_cols)
            
            nerf_depth_weights = render_result['depth_weights'] # (N_select_rows, N_select_cols, N_sample)


            ########
            # First of all, wow, great catch with t_vals_noisy as the new FIXED loss metric for depth!
            # That is an extremely fortunate catch as a consequence of the exploding gradients, which still exist.
            # So, we have discovered that in scenarios where depth loss is a dominant factor to optimize againt,
            # indeed somehow NeRF density field can end up spitting out all 0's entirely...
            # Note the interesting flag here for "start training color network" on top of density network...
            # Naturally, when you multiply a lot of 0's by anything, still 0, then that's the loss?  No!  Problem appears to be NeRF not probabalistic?
            # That is, why isn't the NeRF output for density a uniform distribution over the distance!?
            # This is absolutely, as well, related to the opportunity to implement signed-distance-function loss using the depth supervision...
            # Then, we could potentially decrease number of t_vals, and furthermore, we could put t_vals close to depth! (with some random variance; Gaussian)
            ########




            # collect sensor data and format it
            sensor_depth = scene_train.depths[i,:,:].to(my_devices) # (H, W)
            sensor_depth_selected = sensor_depth[r_id,:][:, c_id] # (N_select_rows, N_select_cols)
            sensor_depth_selected = torch.unsqueeze(sensor_depth_selected, 2) # (N_select_rows, N_select_cols, 1)
            sensor_depth_selected = sensor_depth_selected.expand(-1,-1,args.num_sample) # (N_select_rows, N_select_cols, N_sample)

            # collect sampling distance data and format it
            #t_vals = torch.unsqueeze(torch.unsqueeze(t_vals, 0), 0) # (1,1, N_sample)
            #t_vals = t_vals.expand(args.train_rand_rows, args.train_rand_cols, -1) # (N_select_rows, N_select_cols, N_sample)

            # get a metric for every sampled distance of how far that is from the sensor depth
            squared_difference_weighted_sensor_depth = (sensor_depth_selected - t_vals)**2 # (N_select_rows, N_select_cols, N_sample)

            # go for a Gaussian distribution around the sensor weights
            depth_mean = nerf_depth_weights
            # variance of the NeRF depth values will be random values between 0.0 - 1.0, multiplied by some clamping factor to limit max variance, e.g. 0.01
            # variance helps to prevent system from overfitting to sensor depth data

            min_variance = args.depth_variance * -1
            max_variance = args.depth_variance * 1
            depth_variance = (max_variance - min_variance) * torch.rand((args.train_rand_rows, args.train_rand_cols, args.num_sample), device=c2w.device, dtype=torch.float32) + min_variance # (N_select_rows, N_select_cols, N_sample) 
            resampled_nerf_depths = depth_mean + depth_variance #torch.normal(mean=depth_mean, std=depth_variance)

            # now, take the metric of sensor distance per ray sample and multiply that with NeRF weights: higher loss values will emerge for values further from sensor data
            nerf_sensor_distribution_overlap = squared_difference_weighted_sensor_depth * resampled_nerf_depths # (N_select_rows, N_select_cols, N_sample)

            # take the mean distance value for all of the samples
            mean_depth_loss = torch.mean(nerf_sensor_distribution_overlap)
            log_depth_loss = 1/(-torch.log(mean_depth_loss)) # use 1/(-log(loss)) as a way to preserve gradient signal even for very very small differences
            depth_loss = log_depth_loss / args.empirical_maximum_depth_loss # approximately normalize per-image depth loss between 0 - 1.0


            #depth_loss = F.mse_loss(nerf_depth, sensor_depth_selected)
            rgb_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image
            rgb_loss = 1/(-torch.log(rgb_loss))
            rgb_loss = rgb_loss / args.empirical_maximum_rgb_loss # approximately normalize per-image RGB loss between 0 - 1.0

            ###
            # pose deviation metrics for loss
            ### 
            initial_camera_position = scene_train.c2ws[i][:3,3]
            network_camera_position = c2w[:3,3]
            euclidian_distance_metric = torch.sqrt(((initial_camera_position[0] - network_camera_position[0])**2 + (initial_camera_position[1] - network_camera_position[1])**2 +  (initial_camera_position[2] - network_camera_position[2])**2 ))

            # if epoch_i % eval_image_interval == 0:
            #     print("\n\n---IMAGE {}---".format(scene_train.img_names[i]))
            #     print("\nAverage (x,y,z) translation difference between initial camera position and network camera position: {}".format(euclidian_distance_metric))
            #     print("\nInitial sensor-based camera (x,y,z):\n{}".format(initial_camera_position))
            #     print("\nTrained network camera (x,y,z):\n{}".format(network_camera_position))

            cam_pos_loss = 1/(-torch.log(euclidian_distance_metric))
            position_importance_for_current_epoch = ((number_of_epochs - epoch_i ) / (number_of_epochs ))**2 # exponentially decrease requirement to keep (x,y,z) cam position fixed
            initial_position_loss_importance = args.pose_deviation_loss_initial_importance
            position_loss_importance = initial_position_loss_importance * position_importance_for_current_epoch


            # cam_pos_importance_for_current_epoch = ((number_of_epochs - epoch_i ) / (number_of_epochs )) # linearly decrease dependence of depth as optimization finishes
            # initial_cam_pos_to_rgb_importance = 100.0
            # cam_pos_to_rgb_importance = initial_cam_pos_to_rgb_importance * cam_pos_importance_for_current_epoch # use e^(log(loss)) as a way to preserve gradient signal even for very very small differences

            initial_camera_rotation = scene_train.c2ws[i][:3,:3]
            network_camera_rotation = c2w[:3,:3]
            if epoch_i % eval_image_interval == 0:
                angular_distance_metric = compute_angular_distance(initial_camera_rotation, network_camera_rotation, show_angles=True) # .cpu().detach().numpy() .cpu().detach().numpy()
                # print("\nInitial sensor-based camera 3x3 rotation matrix:\n{}".format(initial_camera_rotation))
                # print("\nTrained network camera 3x3 rotation matrix:\n{}".format(network_camera_rotation))
            else:
                angular_distance_metric = compute_angular_distance(initial_camera_rotation, network_camera_rotation, show_angles=False)

            angular_distance_metric = angular_distance_metric / (math.pi * 2)
            angular_distance_loss = 1 /(-torch.log(angular_distance_metric))
            angular_importance_for_current_epoch = ((number_of_epochs - epoch_i ) / (number_of_epochs ))**2 # exponentially decrease requirement to keep (pitch,yaw,roll) of cam fixed
            initial_angular_loss_importance = args.pose_deviation_loss_initial_importance
            angular_loss_importance = initial_angular_loss_importance * angular_importance_for_current_epoch

            # here, we implement the decaying depth loss supervision, so that images can take priority in inverse rendering, after some time in the optimization
            # depth_loss_exponential_decay_rate = args.depth_loss_exponential_decay_rate # rate between 0.01 - 10.0 to decay the fraction 0

            # x = (number_of_epochs - epoch_i ) / (number_of_epochs)


            depth_importance = polynomial_decay(current_step=epoch_i, 
                                                total_steps=number_of_epochs, 
                                                start_value=args.depth_loss_initial_importance, 
                                                end_value=args.depth_loss_final_importance, 
                                                exponential_index=args.depth_loss_exponential_index, 
                                                curvature_shape=args.depth_loss_curve_shape)


            # exponent = torch.tensor((-1 * x * depth_loss_exponential_decay_rate), device=my_devices)
            # depth_importance_for_current_epoch = torch.exp(exponent) # exponentially decrease dependence of depth as optimization finishes

            # initial_depth_importance = args.depth_loss_initial_importance #100.0
            # depth_importance = initial_depth_importance * depth_importance_for_current_epoch # use e^(log(loss)) as a way to preserve gradient signal even for very very small differences

            # RGB importance will hold strong and increasingly dominate the loss
            rgb_importance = args.rgb_loss_importance #10.0

            # compute final loss metric
            if depth_importance <= 1.0:
                loss = (depth_importance * depth_loss) + ((1 - depth_importance) * rgb_loss) #+ (position_loss_importance * cam_pos_loss) + (angular_loss_importance * angular_distance_loss)
                depth_loss_to_rgb_loss_ratio = depth_importance / (1 - depth_importance)
            else:
                print("System exiting early, as depth importance greater than 1: {}".format(depth_importance))
                print("Depth loss exponential decay rate: {}".format(depth_loss_exponential_decay_rate))
                print("Depth loss importance for current epoch: {}".format(depth_importance_for_current_epoch))
                print("Initial depth importance: {}".format(initial_depth_importance))
                sys.exit(0)


            if epoch_i % eval_image_interval == 0:
                # print("\nPer-Image Depth Loss: {:.5f}, RGB Inverse Render Loss: {:.5f}\n\n".format(depth_loss, rgb_loss))
                per_image_depth_loss["{}".format(scene_train.img_names[i])] = depth_loss
                per_image_rgb_loss["{}".format(scene_train.img_names[i])] = rgb_loss


            if torch.isnan(depth_loss):
                print("Terminating early given NaN in depth loss, debug info below")

                print("\nNeRF depth weights for all samples, first row and col: {}".format(nerf_depth_weights[0,0,:]))
                print("\nSensor depth selected...: {}".format(sensor_depth_selected[0,0,:]))
                print("\nt_vals...: {}".format(t_vals[0,0,:]))
                print("\nsquared_difference_weighted_sensor_depth...: {}".format(squared_difference_weighted_sensor_depth[0,0,:]))
                print("\ndepth_variance...: {}".format(depth_variance[0,0,:]))
                print("\nresampled_nerf_depths...: {}".format(resampled_nerf_depths[0,0,:]))
                print("\nnerf_sensor_distribution_overlap...: {}".format(nerf_sensor_distribution_overlap[0,0,:]))
                print("\nmean depth loss...: {}".format(mean_depth_loss))
                print("\nlog depth loss...: {}".format(log_depth_loss))

                print("NeRF depth weights is NaN: {}".format(torch.isnan(nerf_depth_weights)))
                print("\nSensor depth selected...: {}".format(torch.isnan(sensor_depth_selected)))
                print("\nt_vals...: {}".format(torch.isnan(t_vals)))
                print("\nsquared_difference_weighted_sensor_depth...: {}".format(torch.isnan(squared_difference_weighted_sensor_depth)))
                print("\ndepth_variance...: {}".format(torch.isnan(depth_variance)))
                print("\nresampled_nerf_depths...: {}".format(torch.isnan(resampled_nerf_depths)))
                print("\nnerf_sensor_distribution_overlap...: {}".format(torch.isnan(nerf_sensor_distribution_overlap)))
                print("\nmean depth loss...: {}".format(torch.isnan(mean_depth_loss)))
                print("\nlog depth loss...: {}".format(torch.isnan(log_depth_loss)))
            else:
                print("\nNeRF depth weights for all samples, first row and col: {}".format(nerf_depth_weights[0,0,:]))
                print("\nSensor depth selected...: {}".format(sensor_depth_selected[0,0,:]))
                print("\nt_vals...: {}".format(t_vals[0,0,:]))
                print("\nsquared_difference_weighted_sensor_depth...: {}".format(squared_difference_weighted_sensor_depth[0,0,:]))
                print("\ndepth_variance...: {}".format(depth_variance[0,0,:]))
                print("\nresampled_nerf_depths...: {}".format(resampled_nerf_depths[0,0,:]))
                print("\nnerf_sensor_distribution_overlap...: {}".format(nerf_sensor_distribution_overlap[0,0,:]))
                print("\nmean depth loss...: {}".format(mean_depth_loss))
                print("\nlog depth loss...: {}".format(log_depth_loss))                

            if torch.isnan(rgb_loss):
                print("\n\n--NaN encountered for input image {}, here is the debug summary:--\n\n".format(i))
                print("The camera2world matrix (NaN inside: {}):\n{}\n".format(torch.isnan(c2w).any(), c2w))
                print("The focal length (NaN inside: {}):\n{}\n".format(torch.isnan(fxfy).any(), fxfy))
                print("The ray direction of the camera (NaN inside: {}):\n{}\n".format(torch.isnan(ray_dir_cam).any(), ray_dir_cam))
                print("The r_id (NaN inside: {}):\n{}\n".format(torch.isnan(r_id).any(), r_id))
                print("The c_id (NaN inside: {}):\n{}\n".format(torch.isnan(c_id).any(), c_id))
                print("The ray_selected_cam (NaN inside: {}):\n{}\n".format(torch.isnan(ray_selected_cam).any(), ray_selected_cam))
                print("The img_selected (NaN inside: {}):\n{}\n".format(torch.isnan(img_selected).any(), img_selected))
                print("The render_result - RGB (NaN inside: {}):\n{}\n".format(torch.isnan(render_result['rgb']).any(), render_result['rgb']))
                print("The render_result - Depth (NaN inside: {}):\n{}\n".format(torch.isnan(render_result['depth_map']).any(), render_result['depth_map']))
                print("And the L2 loss: {}".format(rgb_loss))
            else:
                loss.backward()

                optimizer_nerf.step()
                optimizer_focal.step()
                optimizer_pose.step()

                optimizer_nerf.zero_grad()
                optimizer_focal.zero_grad()
                optimizer_pose.zero_grad()

                weighted_sum_of_losses = weighted_sum_of_losses + loss
                sum_of_losses = sum_of_losses + rgb_loss + depth_loss #+ cam_pos_loss + angular_distance_loss
                sum_of_rgb_losses = sum_of_rgb_losses + rgb_loss
                sum_of_depth_losses = sum_of_depth_losses + depth_loss
                sum_of_cam_pos_losses = sum_of_cam_pos_losses + cam_pos_loss
                sum_of_cam_rot_losses = sum_of_cam_rot_losses + angular_distance_loss

                sum_of_depth_to_rgb_loss_ratios = sum_of_depth_to_rgb_loss_ratios + depth_loss_to_rgb_loss_ratio

                count_of_losses = count_of_losses + 1

    average_weighted_loss = torch.div(weighted_sum_of_losses, count_of_losses)
    average_loss = torch.div(sum_of_losses, count_of_losses)
    average_rgb_loss = torch.div(sum_of_rgb_losses, count_of_losses)
    average_depth_loss = torch.div(sum_of_depth_losses, count_of_losses)
    average_cam_pos_loss = torch.div(sum_of_cam_pos_losses, count_of_losses)
    average_cam_rot_loss = torch.div(sum_of_cam_rot_losses, count_of_losses)
    average_depth_to_rgb_loss_ratio = torch.div(sum_of_depth_to_rgb_loss_ratios, count_of_losses)


    if epoch_i % args.log_frequency == 0:
        wandb.log({"Weighted_Loss": average_weighted_loss,
                   "Unweighted_Loss": average_loss,
                   "RGB Inverse Render Loss": average_rgb_loss,
                   "Depth Loss": average_depth_loss,
                   "Ratio of Depth Loss to RGB Loss": average_depth_to_rgb_loss_ratio,
                   "(x,y,z) camera position deviation + (pitch,yaw,roll) camera perspective deviation": average_cam_pos_loss + average_cam_rot_loss,
                   })

    if epoch_i % args.log_frequency == 0:
        print("({}) RGB Loss: {:.5f}, Depth Loss: {:.5f}, Depth to RGB Loss Ratio: {:.3f}, Unweighted Loss: {:.5f}".format(epoch_i, average_rgb_loss, average_depth_loss, average_depth_to_rgb_loss_ratio, average_loss))

    #L2_loss_epoch.append(L2_loss.item())
    #L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.

    mean_losses = {
        'L2': weighted_sum_of_losses.item(),
    }
    return mean_losses


def main(args):

    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    exp_root_dir = Path(os.path.join('./logs/nerfmm', args.scene_name))
    exp_root_dir.mkdir(parents=True, exist_ok=True)




    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy('./models/nerf_models.py', experiment_dir)
    shutil.copy('./models/intrinsics.py', experiment_dir)
    shutil.copy('./models/poses.py', experiment_dir)
    shutil.copy('./tasks/nerfmm/train.py', experiment_dir)

    if args.store_pose_history:
        pose_history_dir = Path(os.path.join(experiment_dir, 'pose_history'))
        pose_history_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(args)

    '''Summary Writer'''
    writer = SummaryWriter(log_dir=str(experiment_dir))

    '''Data Loading'''
    scene_train = DataLoaderWithSmartphone( base_dir=args.base_dir,
                                            scene_name=args.scene_name,
                                            data_type='train',
                                            res_ratio=args.resize_ratio,
                                            num_img_to_load=args.train_img_num,
                                            skip=args.train_skip,
                                            use_ndc=args.use_ndc)

    # The COLMAP eval poses are not in the same camera space that we learned so we can only check NVS
    # with a 4x4 identity pose.
    # eval_c2ws = torch.eye(4).unsqueeze(0).float()  # (1, 4, 4)

    evaluation_image_index = 0
    print("Evaluation image is {}".format(scene_train.img_names[evaluation_image_index]))

    print('Train with {0:6d} images.'.format(scene_train.imgs.shape[0]))

    '''Model Loading'''
    pos_enc_in_dims = (2 * args.pos_enc_levels + int(args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    if args.use_dir_enc:
        dir_enc_in_dims = (2 * args.dir_enc_levels + int(args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    else:
        dir_enc_in_dims = 0

    model = OfficialNerf(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims)
    wandb.watch(model)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)

    # learn focal parameter
    if args.start_refine_focal_epoch > -1:
        focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only,
                               order=args.focal_order, init_focal=scene_train.focal)
    else:
        focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)

    # learn pose for each image
    if args.start_refine_pose_epoch > -1:
        pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, scene_train.c2ws)
    else:
        pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)

    # Work In Progress: "PoseNet" a network to learn 6D pose transformations without explicitly updating 6D pose parameters per image.  
    # pose_param_net = PoseNet(args.learn_R, args.learn_t, scene_train.c2ws, args.hidden_dims, scene_train.camera_position_encodings, scene_train.camera_direction_encodings)

    # Original implementation has support for multi-GPU.
    # if args.multi_gpu:
    #     pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    # else:
    pose_param_net = pose_param_net.to(device=my_devices)

    '''Set Optimiser'''
    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=args.nerf_lr)
    optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=args.focal_lr)
    optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=args.pose_lr)

    scheduler_nerf = torch.optim.lr_scheduler.MultiStepLR(optimizer_nerf, milestones=args.nerf_milestones,
                                                          gamma=args.nerf_lr_gamma)
    scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(optimizer_focal, milestones=args.focal_milestones,
                                                           gamma=args.focal_lr_gamma)
    scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=args.pose_milestones,
                                                          gamma=args.pose_lr_gamma)

    '''Training'''
    for epoch_i in range(args.epoch):
        rgb_act_fn = torch.sigmoid
        train_epoch_losses = train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose,
                                             model, focal_net, pose_param_net, my_devices, args, rgb_act_fn, epoch_i)
        train_L2_loss = train_epoch_losses['L2']
        scheduler_nerf.step()
        scheduler_focal.step()
        scheduler_pose.step()

        # train_psnr = mse2psnr(train_L2_loss)
        # writer.add_scalar('train/mse', train_L2_loss, epoch_i)
        # writer.add_scalar('train/psnr', train_psnr, epoch_i)
        # writer.add_scalar('train/lr', scheduler_nerf.get_lr()[0], epoch_i)
        # logger.info('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))
        # tqdm.write('{0:6d} ep: Train: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, train_L2_loss, train_psnr))

        # pose_history_milestone = list(range(0, 100, 5)) + list(range(100, 1000, 100)) + list(range(1000, 10000, 1000))
        # if epoch_i in pose_history_milestone:
        #     with torch.no_grad():
        #         if args.store_pose_history:
        #             store_current_pose(pose_param_net, pose_history_dir, epoch_i)

        # if epoch_i % args.eval_cam_interval == 0 and epoch_i > 0:

            # with torch.no_grad():
            #     eval_stats_tran, eval_stats_rot, eval_stats_scale = eval_one_epoch_traj(scene_train, pose_param_net)
            #     writer.add_scalar('eval/traj/translation', eval_stats_tran['mean'], epoch_i)
            #     writer.add_scalar('eval/traj/rotation', eval_stats_rot['mean'], epoch_i)
            #     writer.add_scalar('eval/traj/scale', eval_stats_scale['mean'], epoch_i)

            #     logger.info('{0:6d} ep Traj Err: translation: {1:.6f}, rotation: {2:.2f} deg, scale: {3:.2f}'.format(epoch_i,
            #                                                                                                          eval_stats_tran['mean'],
            #                                                                                                          eval_stats_rot['mean'],
            #                                                                                                          eval_stats_scale['mean']))
            #     print('{0:6d} ep Traj Err: translation: {1:.6f}, rotation: {2:.2f} deg, scale: {3:.2f}'.format(epoch_i,
            #                                                                                                         eval_stats_tran['mean'],
            #                                                                                                         eval_stats_rot['mean'],
            #                                                                                                         eval_stats_scale['mean']))

            #     fxfy = focal_net(0)
            #     print('Est fx: {0:.2f}, fy {1:.2f}, COLMAP focal: {2:.2f}'.format(fxfy[0].item(), fxfy[1].item(),
            #                                                                            scene_train.focal))
            #     logger.info('Est fx: {0:.2f}, fy {1:.2f}, COLMAP focal: {2:.2f}'.format(fxfy[0].item(), fxfy[1].item(),
            #                                                                             scene_train.focal))
            #     if torch.is_tensor(fxfy):
            #         L1_focal = torch.abs(fxfy - scene_train.focal).mean().item()
            #     else:
            #         L1_focal = np.abs(fxfy - scene_train.focal).mean()
            #     writer.add_scalar('eval/L1_focal', L1_focal, epoch_i)

        if epoch_i % args.eval_img_interval == 0 and epoch_i > 0:
            with torch.no_grad():
                evaluation_image_pose = pose_param_net(evaluation_image_index)

                evaluation_image_pose = evaluation_image_pose.unsqueeze(0).float() # (1, 4, 4)


                eval_one_epoch_img(evaluation_image_pose, scene_train, model, focal_net, pose_param_net, my_devices,
                                   args, epoch_i, writer, rgb_act_fn, evaluation_image_index)

                # # save the latest model.
                # save_checkpoint(epoch_i, model, optimizer_nerf, experiment_dir, ckpt_name='latest_nerf')
                # save_checkpoint(epoch_i, focal_net, optimizer_focal, experiment_dir, ckpt_name='latest_focal')
                # save_checkpoint(epoch_i, pose_param_net, optimizer_pose, experiment_dir, ckpt_name='latest_pose')
    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    main(args)
