import torch
import torch.nn as nn
from utils.lie_group_helper import make_c2w


class PoseNet(nn.Module):
    def __init__(self, learn_R, learn_t, init_c2w, D, camera_position_encodings, camera_direction_encodings):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(PoseNet, self).__init__()

        # save the one-and-done encodings of the camera
        self.camera_position_encodings = camera_position_encodings
        self.camera_direction_encodings = camera_direction_encodings

        # get size of encodings for position and direction from actual data
        pos_in_dims = camera_position_encodings.shape[1]
        dir_in_dims = camera_direction_encodings.shape[1]

        self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        # self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        # self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU()
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU()
        )

        self.fc_pose_translation = nn.Linear(D, 3) # pose translation in 3D space

        self.fc_feature = nn.Linear(D, D)
        self.directional_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_pose_rotation = nn.Linear(D//2, 3)

    def forward(self, cam_id):
        cam_position_encoding = self.camera_position_encodings[cam_id] # (pos_in_dims)
        cam_direction_encoding = self.camera_direction_encodings[cam_id] # (dir_in_dims)

        cam_position_encoding = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(cam_position_encoding, 0), 0), 0) # (1, 1, 1, pos_in_dims)
        cam_direction_encoding = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(cam_direction_encoding, 0), 0), 0) # (1, 1, 1, dir_in_dims)

        x = self.layers0(cam_position_encoding) # (1, 1, 1, D)
        x = torch.cat([x, cam_position_encoding], dim=3)  # (1, 1, N_sample, D + pos_in_dims)
        x = self.layers1(x)  # (1, 1, 1, D)

        translation = self.fc_pose_translation(x)  # (1, 1, 1, 3)

        feat = self.fc_feature(x)  # (1, 1, 1, D)
        x = torch.cat([feat, cam_direction_encoding], dim=3)  # (1, 1, 1, D + dir_in_dims)
        x = self.directional_layers(x)  # (1, 1, 1, D/2)
        rotation = self.fc_pose_rotation(x)  # (1, 1, 1, 3) in axis-angle representation

        translation = torch.squeeze(translation)
        rotation = torch.squeeze(rotation)

        # print("Final translation output by network: {}".format(translation))
        # print("Final rotation output by network: {}".format(rotation))

        c2w = make_c2w(rotation, translation)  # (4, 4)

        # print("Made c2w: {}".format(c2w))

        # learn a delta pose between init pose and target pose, if a init pose is provided
        c2w = c2w @ self.init_c2w[cam_id]

        # print("Final c2w: {}".format(c2w))

        return c2w
