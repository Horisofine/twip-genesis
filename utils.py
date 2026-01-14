import torch


def quat_multiply(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.

    Parameters:
    q (torch.Tensor): A tensor of shape (..., 4) representing the first quaternion (w, x, y, z).
    r (torch.Tensor): A tensor of shape (..., 4) representing the second quaternion (w, x, y, z).

    Returns:
    torch.Tensor: A tensor of shape (..., 4) representing the product quaternion.
    """
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def quat_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to roll, pitch, and yaw angles.

    Parameters:
    quat (torch.Tensor): A tensor of shape (..., 4) representing the quaternions (w, x, y, z).

    Returns:
    torch.Tensor: A tensor of shape (..., 3) representing the roll, pitch, and yaw angles in radians.
    """
    w, x, y, z = quat.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1,
                        torch.sign(sinp) * (torch.pi / 2),
                        torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def rotate_quat_yaw(quat: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Rotate a quaternion by a given yaw angle.

    Parameters:
    quat (torch.Tensor): A tensor of shape (..., 4) representing the quaternions (w, x, y, z).
    yaw (torch.Tensor): A tensor of shape (...) representing the yaw angles in radians.

    Returns:
    torch.Tensor: A tensor of shape (..., 4) representing the rotated quaternions.
    """
    # Quaternion format: (w, x, y, z)
    half_angle = yaw / 2
    cos_half = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)

    quat_yaw = torch.stack([cos_half, torch.zeros_like(cos_half), sin_half, torch.zeros_like(cos_half)], dim=-1)

    return quat_multiply(quat_yaw, quat)

def rotate_quat_pitch(quat: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """
    Rotate a quaternion by a given pitch angle.

    Parameters:
    quat (torch.Tensor): A tensor of shape (..., 4) representing the quaternions (w, x, y, z).
    pitch (torch.Tensor): A tensor of shape (...) representing the pitch angles in radians.

    Returns:
    torch.Tensor: A tensor of shape (..., 4) representing the rotated quaternions.
    """
    # Quaternion format: (w, x, y, z)
    half_angle = pitch / 2
    cos_half = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)

    quat_pitch = torch.stack([cos_half, sin_half, torch.zeros_like(cos_half), torch.zeros_like(cos_half)], dim=-1)

    return quat_multiply(quat_pitch, quat)

def rotate_quat_roll(quat: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """
    Rotate a quaternion by a given roll angle.

    Parameters:
    quat (torch.Tensor): A tensor of shape (..., 4) representing the quaternions (w, x, y, z).
    roll (torch.Tensor): A tensor of shape (...) representing the roll angles in radians.

    Returns:
    torch.Tensor: A tensor of shape (..., 4) representing the rotated quaternions.
    """
    # Quaternion format: (w, x, y, z)
    half_angle = roll / 2
    cos_half = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)

    quat_roll = torch.stack([cos_half, torch.zeros_like(cos_half), torch.zeros_like(cos_half), sin_half], dim=-1)

    return quat_multiply(quat_roll, quat)