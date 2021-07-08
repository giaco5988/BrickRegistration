import os
import time
import glob
import random
import math
from pathlib import Path
import logging

from PIL import Image
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


def render_generator(mode, colors, all_urdf_with_class_id, rng, nb_obj, nb_view, out_folder, scene_name):
    """"""
    p.connect(mode)
    # reset la simulation
    p.resetSimulation()

    # reduce timeStep if the physics become unstable
    # but you will need to take more steps before the system reach equilibrium
    time_step = 0.01
    p.setTimeStep(time_step)

    # PyBullet_data package
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # load file plane.urdf (URDF = Unified Robot Description Format)
    plane_id = p.loadURDF("plane.urdf")

    objs = [plane_id]
    mesh_scale = [0.1, 0.1, 0.1]
    collision_dict = {}
    seg_id_to_class = [0]

    for i in tqdm(range(nb_obj), total=nb_obj, desc=f'Load objects scene {scene_name}'):
        cube_start_pos = [rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(10, 50)]
        cube_start_orientation = p.getQuaternionFromEuler(
            [rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)])
        (obj_name, obj_id) = rng.choice(all_urdf_with_class_id)
        try:
            LOGGER.debug("loading object " + str(i) + " / " + str(nb_obj))
            LOGGER.debug(obj_name)
            # we don't use loadURDF to be able to change the color of the bricks easily
            # objs.append( p.loadURDF(objname,cubeStartPos,cubeStartOrientation ))

            # When we create multiple instances of the same object
            # there are some performance gain to share the shapes between rigid bodies
            # Instead of p.loadURDF you can use the following
            name = Path(obj_name).name
            name_root = os.path.splitext(str(name))[0]
            partition = os.path.dirname(obj_name)
            color = rng.choice(colors)
            filename = os.path.join(partition, name_root, f"{name_root}.obj")
            vs = p.createVisualShape(p.GEOM_MESH, fileName=filename, meshScale=mesh_scale, rgbaColor=color)

            if name in collision_dict:
                cuid = collision_dict[name]
            else:
                filename = os.path.join(partition, name_root, f"{name_root}_vhacd.obj")
                cuid = p.createCollisionShape(p.GEOM_MESH, fileName=filename, meshScale=mesh_scale)
                collision_dict[name] = cuid
            objs.append(p.createMultiBody(1.0, cuid, vs, cube_start_pos, cube_start_orientation))
            seg_id_to_class.append(obj_id)
        except p.error:
            LOGGER.warning("failed to load : " + obj_name)

    p.setGravity(0, 0, -10)

    nbsteps = 500
    for i in range(nbsteps):
        p.stepSimulation()
        LOGGER.debug("step " + str(i) + " / " + str(nbsteps))

    for j in tqdm(range(nb_view), total=nb_view, desc=f'Render view scene {scene_name}'):
        LOGGER.debug("rendering view " + str(j))
        ang = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(1, 30)
        h = rng.uniform(50, 100)
        fov = 45
        aspect = 1
        near_val = 0.1
        far_val = 150.1

        cam_params = [r*math.cos(ang), r*math.sin(ang), h, 0, 0, 0, 0, 0, 1, fov, aspect, near_val, far_val]

        LOGGER.debug("camParams", cam_params)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_params[0:3],
            cameraTargetPosition=cam_params[3:6],
            cameraUpVector=cam_params[6:9])

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=cam_params[9],
            aspect=cam_params[10],
            nearVal=cam_params[11],
            farVal=cam_params[12])

        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=1080,
            height=1080,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix)

        Path(out_folder).mkdir(parents=True, exist_ok=True)
        view_str = "-view"+str(j)
        rgb_array = np.reshape(rgb_img, (height, width, 4)).astype(np.uint8)
        im = Image.fromarray(rgb_array)
        im.save(os.path.join(out_folder, scene_name + view_str + ".png"))

        # We also compute here information which will be interesting to try to predict
        # Typically we are going to ask a neural network to predict
        # or each pixel
        # the offset to the centroid/center of the object the pixel belongs to
        # For each pixel the NN is making a prediction of which class it belongs to
        # For each pixel the NN is making a prediction of the orientation of the object belongs to
        # For each pixel the NN is making a prediction of the 3D bounding box coordinates the object belongs to
        # ...
        # Then all you need is to run a clustering algorithm like DBSCAN or HDBSCAN on the neural network prediction

        # In pixel space the 2D centroid of all the visible pixels of the object
        # if no pixel from the object is visible in the image the row will contain nan
        centroids = np.array([np.mean(np.argwhere(seg_img == idx), axis=0) for idx in range(len(objs))])

        # In absolute world coordinates
        position_and_orientation = [p.getBasePositionAndOrientation(idx) for idx in objs]
        pos = np.array([x[0] for x in position_and_orientation])
        orient = np.array([x[1] for x in position_and_orientation])

        # We compute the coordinates of the center of the object in screen Coordinates
        p_m = np.reshape(projection_matrix, (4, 4))
        v_m = np.reshape(view_matrix, (4, 4))
        pos1 = np.hstack([pos, np.ones((np.shape(pos)[0], 1))])
        xyzw = np.dot(np.dot(pos1, v_m), p_m)
        pos_in_ndc = xyzw[:, 0:3] / xyzw[:, 3:]
        view_size = np.reshape(np.array([width, height], dtype=np.float32), (1, 2))
        view_offset = view_size / 2.0 - 0.5
        pos_in_screen_coordinates = pos_in_ndc[:, 0:2] * view_size + view_offset
        depth_in_screen_coordinates = pos_in_ndc[:, 2]

        # We can do the same technique for each corner of the 3d bounding box of the object
        # ...

        # For the orientation of the object in view coordinates
        rv = Rotation.from_matrix(v_m[0:3, 0:3])
        orientation_in_view_coordinates = np.array([(rv * Rotation.from_quat(orient[i])).as_quat()
                                                    for i in range(orient.shape[0])])

        # We don't allow_pickle for safety reasons
        np.savez(os.path.join(out_folder, "parameters_" + scene_name + view_str),
                 camParams=np.array(cam_params),
                 position=pos,
                 orientation=orient,
                 centroids=centroids,
                 posInScreenCoordinates=pos_in_screen_coordinates,
                 orientationInViewCoordinates=orientation_in_view_coordinates,
                 depthInScreenCoordintes=depth_in_screen_coordinates,
                 segImg=seg_img,
                 segIdToClass=np.array(seg_id_to_class),
                 allow_pickle=False)

        # don't forget to put the allow_pickle=False in the load (that's where it matters!)
        # with np.load(outFolder + "/" + "parameters" + sceneName + viewStr + ".npz",allow_pickle=False ) as data:
        #    print("npz keys :")
        #    for k in data.files:
        #        print(k)
        #        print(data[k])

        # We don't store output rasterized images as they will be generate on the fly from the segmentation map
        # With something like the following
        # classImg = np.array(segIdToClass)[ segImg.reshape((-1,))].reshape(segImg.shape)
        # The offsets in pixel space will have to be computed on the fly to from the absolute coordinates in pixel space

    # instead of stepping manually you can step in real time,
    # but be careful that you don't have too many objects or it may become unstable
    # p.setRealTimeSimulation(1)

    if mode == p.GUI:
        while True:
            time.sleep(0.01)
            p.stepSimulation()

    p.disconnect()


def test():
    """"""
    # use mode = p.DIRECT for usage without gui
    # mode = p.GUI
    mode = p.DIRECT
    # In GUI mode it became slow at around 500 objects
    # In Direct mode, I could run it OK with 3000 similar object
    nb_colors = 3

    rng = random.Random(42)
    colors = [[rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), 1.0] for i in range(nb_colors)]

    # we exclude all files beginning by _ for example _prototype.urdf
    all_urdf = sorted(glob.glob("**/*lego-*/[!_]*.urdf"))
    LOGGER.info("Number of urdf :" + str(len(all_urdf)))

    nb_obj = 100
    nb_view = 10

    # We reserve class 0 for background
    all_urdf_with_class_id = [(all_urdf[i], i + 1) for i in range(len(all_urdf))]
    rng = random.Random(42)
    render_generator(mode, colors, all_urdf_with_class_id, rng, nb_obj, nb_view, "Renderings", "Scene0")

    # It 's deterministic and we get the same results
    rng = random.Random(42)
    render_generator(mode, colors, all_urdf_with_class_id, rng, nb_obj, nb_view, "Renderings", "Scene0bis")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test()
