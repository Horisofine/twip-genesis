import genesis as gs
import numpy as np

gs.init(backend=gs.cuda)

scene = gs.Scene(
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
twip = scene.add_entity(
    gs.morphs.URDF(
        file="assets/twip.urdf",
        pos=[0, 0, 0.1]
    )
)

joints_name = [
    "rwheel",
    "lwheel",
]

motors_dof_idx = [twip.get_joint(name).dof_idx_local for name in joints_name]

scene.build()
flip = True

for i in range(1000):
    if i % 50 == 0:
        flip = not flip

    if flip:
        velocity = np.array([10.0, 10.0])
    else:
        velocity = np.array([-10.0, -10.0])

    twip.control_dofs_velocity(
        velocity,
        motors_dof_idx,
    )
    scene.step()