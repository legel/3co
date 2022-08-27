import torch
import sys

def volume_sampling(poses, pixel_directions, sampling_depths, perturb_depths=True):
    # poses := (N_pixels, 4, 4)
    # pixel_directions := (N_pixels, 3)
    # sampling_depths := (N_pixels, N_samples)

    N_pixels = pixel_directions.shape[0]
    N_samples = sampling_depths.shape[1]

    # transform rays from camera coordinate to world coordinate
    pixel_directions_world = torch.matmul(poses[:,:3, :3], pixel_directions.unsqueeze(2)).squeeze(2)  # (N, 3, 3) * (N, 3, 1) -> (N, 3) .squeeze(3) 

    poses_xyz = poses[:, :3, 3]  # the translation vectors (N, 3)

    # this perturb only works if we sample depth linearly, not the disparity.
    if perturb_depths:
        # add some noise to each each z_val
        near = torch.min(sampling_depths)
        far = torch.max(sampling_depths)
        depth_noise = torch.rand((N_pixels, N_samples), device=poses.device, dtype=torch.float32)  # (N_pixels, N_samples)
        depth_noise = depth_noise * (far - near) / N_samples # (N_pixels, N_samples)
        resampled_depths = sampling_depths.view(1, N_samples) + depth_noise  # (N_pixels, N_samples)
    else:
        resampled_depths = sampling_depths #sampling_depths.view(1, N_samples).expand(N_pixels, N_samples)

    pixel_depth_samples_world_directions = pixel_directions_world.unsqueeze(1) * resampled_depths.unsqueeze(2) # (N_pixels, N_samples, 3)
    pixel_xyz_positions = poses_xyz.unsqueeze(1).expand(N_pixels, N_samples, 3) + pixel_depth_samples_world_directions # (N_pixels, N_samples, 3)

    return pixel_xyz_positions, pixel_directions_world, resampled_depths # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)


def volume_rendering(rgb, density, depths):
    """
    :param rgb:     (N_pixels, N_sample, 3)     RGB network output
    :param density: (N_pixels, N_sample, 1)     pixel density output
    :param depths:  (N_pixels, N_sample)        pixel density output

    :return:                (N_pixels, 3)               rendered rgb image
                            (N_pixels, N_sample)        weights at each sample position
    """
    N_pixels, N_samples = depths.shape[0], depths.shape[1]

    rgb = torch.sigmoid(rgb)
    density = torch.squeeze(density.relu(), dim=2)  # (N_pixels, N_sample)

    # Compute distances between samples.
    # 1. compute the distances among first (N-1) samples
    # 2. the distance between the LAST sample and infinite far is 1e10
    dists = depths[:, 1:] - depths[:, :-1]  # (N_pixels, N_sample-1)
    dist_far = torch.empty(size=(N_pixels, 1), dtype=torch.float32, device=dists.device).fill_(1e10)  # (H, W, 1)
    dists = torch.cat([dists, dist_far], dim=1)  # (N_pixels, N_sample)

    alpha = 1 - torch.exp(-1.0 * density * dists)  # (N_pixels, N_sample)

    # 1. We expand the exp(a+b) to exp(a) * exp(b) for the accumulated transmittance computing.
    # 2. For the space at the boundary far to camera, the alpha is constant 1.0 and the transmittance at the far boundary
    # is useless. For the space at the boundary near to camera, we manually set the transmittance to 1.0, which means
    # 100% transparent. The torch.roll() operation simply discards the transmittance at the far boundary.
    acc_transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=1)  # (N_pixels, N_sample)
    acc_transmittance = torch.roll(acc_transmittance, shifts=1, dims=1)  # (N_pixels, N_sample)
    acc_transmittance[:, 0] = 1.0  # (N_pixels, N_sample)

    weight = acc_transmittance * alpha  # (N_pixels, N_sample)

    # (N_pixels, N_sample, 1) * (N_pixels, N_sample, 3) = (N_pixels, N_sample, 3) -> (N_pixels, 3)
    rgb_rendered = torch.sum(weight.unsqueeze(2) * rgb, dim=1)
    depth_map = torch.sum(weight * depths, dim=1)  # (N_pixels)

    result = {
        'rgb_rendered': rgb_rendered,  # (N_pixels, 3)
        'weight': weight,  # (N_pixels, N_sample)
        'density': density, # (N_pixels, N_sample)))
        'depth_map': depth_map,  # (N_pixels)
        'alpha': alpha,
        'acc_transmittance': acc_transmittance,
        'rgb': rgb,
        'distances': dists,
    }
    return result