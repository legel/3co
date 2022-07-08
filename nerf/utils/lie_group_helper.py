import numpy as np
import torch
from scipy.spatial.transform import Rotation as RotLib
from ahrs import Quaternion, DCM
from ahrs.utils import angular_distance
# from pytorch3d.transforms.so3 import so3_relative_angle

import math
from typing import Tuple

import torch
import sys

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4

def to_euler(rotation_matrix, rotation_order = "xzy"):
    # outputs Euler angles, basically "pitch", "yaw", "roll"
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html#r72d546869407-3
    return r.as_euler(rotation_order, degrees=True)

def from_euler(euler_angles, rotation_order = 'xzy'):
    # convert from Euler angles to 3x3 rotation matrix
    # Euler angles is a list of size 3 of angles (pitch, yaw, roll)
    return r.from_euler(rotation_order, euler_angles, degrees=True)




def compute_angular_distance(rot_a, rot_b, show_angles):
    # compute angular distance between 2 rotation matrices, each of which is a 3x3 representation
    # rot_a_quat = Quaternion(SO3_to_quat(rot_a))
    # rot_b_quat = Quaternion(SO3_to_quat(rot_b))

    # rot_a_dcm = rot_a_quat.to_DCM()
    # rot_b_dcm = rot_b_quat.to_DCM()

    # angular_distance_metric = angular_distance(rot_a_dcm, rot_b_dcm)

    # print(rot_a)
    # print(rot_b)

    rot_a = torch.unsqueeze(rot_a,0)
    rot_b = torch.unsqueeze(rot_b,0)

    angular_distance_metric = torch.squeeze(so3_relative_angle(R1=rot_a, R2=rot_b, cos_angle=False, cos_bound=0.0000001, eps=0.0000001))
    angular_distance_metric = torch.sqrt(angular_distance_metric * angular_distance_metric)

    # if show_angles:
    #     print("\nAngular distance metric of {}".format(angular_distance_metric))

    return angular_distance_metric

def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`
    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.
    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.
    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)


def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.
    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.
    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
    #     raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.
    More specifically:
    ```
    bounds=(lower_bound, upper_bound)
    if lower_bound <= x <= upper_bound:
        acos_linear_extrapolation(x) = acos(x)
    elif x <= lower_bound: # 1st order Taylor approximation
        acos_linear_extrapolation(x)
            = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
    else:  # x >= upper_bound
        acos_linear_extrapolation(x)
            = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)
    ```
    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)









def SO3_to_quat(R):
    """
    :param R:  (N, 3, 3) or (3, 3) np
    :return:   (N, 4, ) or (4, ) np
    """
    x = RotLib.from_matrix(R)
    quat = x.as_quat()
    return quat


def quat_to_SO3(quat):
    """
    :param quat:    (N, 4, ) or (4, ) np
    :return:        (N, 3, 3) or (3, 3) np
    """
    x = RotLib.from_quat(quat)
    R = x.as_matrix()
    return R


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w


def vec2skew_batch(v):
    """
    :param v:  (N, 3, ) torch tensor
    :return:   (N, 3, 3)
    """
    number_of_samples = v.shape[0]
    zero = torch.zeros((number_of_samples,1), dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[:,2:3],   v[:,1:2]], dim=1)  # (N, 3, 1)
    skew_v1 = torch.cat([ v[:,2:3],   zero,    -v[:,0:1]], dim=1)  # (N, 3, 1)
    skew_v2 = torch.cat([-v[:,1:2],   v[:,0:1],   zero], dim=1)    # (N, 3, 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=2)       # (N, 3, 3)

    return skew_v  # (N, 3, 3)


def Exp_batch(r):
    """so(3) vector to SO(3) matrix
    :param r: (N, 3, ) axis-angle, torch tensor
    :return:  (N, 3, 3)
    """
    batch_size = r.shape[0]

    skew_r = vec2skew_batch(r)  # (N, 3, 3)
    norm_r = r.norm() + 1e-15

    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    reshaped_eye = eye.reshape((1, 3, 3))
    batch_eye = reshaped_eye.repeat(batch_size, 1, 1)

    R = batch_eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)

    return R


def make_pose(r, t):
    """
    :param r:  (N, 3, ) axis-angle             torch tensor
    :param t:  (N, 3, ) translation vector     torch tensor
    :return:   (N, 4, 4)
    """
    R = Exp_batch(r)  # (N, 3, 3)
    pose = torch.cat([R, t.unsqueeze(2)], dim=2)  # (N, 3, 4)
    pose = convert3x4_4x4(pose)  # (N, 4, 4)

    return pose

