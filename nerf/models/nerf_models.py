import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from utils.pos_enc import get_number_of_encoded_dimensions


class NeRFDensity(nn.Module):
    def __init__(self, args):
        super(NeRFDensity, self).__init__()

        pos_in_dims = get_number_of_encoded_dimensions(number_of_fourier_frequencies=args.positional_encoding_fourier_frequencies)
        D = args.density_neural_network_parameters

        self.pos_in_dims = pos_in_dims

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU()
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut nn.BatchNorm2d(D)
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU()
        )


        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.fc_density.bias.data = torch.tensor([0.1]).float()

    def forward(self, pos_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, 4)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=2)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)
        density = self.fc_density(x)  # (H, W, N_sample, 1)
        features = self.fc_feature(x)  # (H, W, N_sample, D)
        return density, features


class NeRFColor(nn.Module):
    def __init__(self, args):
        super(NeRFColor, self).__init__()

        dir_in_dims = get_number_of_encoded_dimensions(number_of_fourier_frequencies=args.directional_encoding_fourier_frequencies)
        self.dir_in_dims = dir_in_dims

        D_color = args.color_neural_network_parameters
        D_density = args.density_neural_network_parameters

        self.rgb_layers = nn.Sequential(
            nn.Linear(D_density + dir_in_dims, D_color//2), nn.ReLU()
            )
        self.fc_rgb = nn.Linear(D_color//2, 3)

    def forward(self, feat, dir_enc, rgb_image):
        """
        :param feat: # (H, W, N_sample, D) features from density network
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb (H, W, N_sample, 3)
        """

        x = torch.cat([feat, dir_enc], dim=2)  # (N_pixels, N_sample, D+dir_in_dims)
        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)  # (H, W, N_sample, 3)

        return rgb