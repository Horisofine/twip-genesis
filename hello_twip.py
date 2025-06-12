import genesis as gs

gs.init(backend=gs.metal)

scene = gs.Scene()

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.URDF(
        file="assets/twip.urdf"
    )
)

scene.build()
for i in range(1000):
    scene.step()