
# Goals for today with Blender rendering and scripting:

# A. Build a Python API (examples below) that can receive commands to setup animation keyframes, with the following incorporated:
# (1) Positions of Iris: (x,y,z,pitch,yaw)
# (2) Positions of Blender Camera: (x,y,z,pitch,yaw)
# (3) Structured Light Sequences: (start_frame, end_frame) 

# B. Ensure rendering is as close to photorealistic as possible, in the iris.blend file:
# (1) Set up HDR environment map (we may eventually do something a little different to increase realism, but for now this hack should help)
# (2) Double check materials are properly imported and applied (Lance will clean-up and fine-tune later)
# (3) Blender Camera is rendering at a ratio of pixel_width:pixel_height equal to 16:9, e.g. 1600:900 pixels.
# (4) Consider adding a few tricks to increase realism of renders, e.g. small depth of field of camera so there is focus blur

# C. Test on an animation with a few frames:
# (1) Ensure .png files are created (e.g. in Blender's /tmp/ directory, where sometimes animation renders get written to)
# (2) Make sure they follow the animation path and that they look perfect

import math
import bpy

fps_resolution = 0.2

class Iris():
# create an Iris class that simplifies the API for working with Blender
# include as attributes:
    def __init__(self, x=0.0, y=-1.5, z=0, pitch=0.0, yaw=0.0):         ## camera is facing forward and is as low as possible
        # position attributes LOCALLY 
        self.x = x # x :: movement: (left,right)  orientation : (+,-) min: -1.3 max: 1.2
        self.y = y # y :: movement: (up, down)    orientation : (+,-) min: -1.5 max: 0
        self.z = z # z :: movement: (front, back) orientation : (-,+) min: -1.3 max: 1.2
        self.pitch = pitch # min: -90° max: 45
        self.yaw = yaw # min: -135° max: 135°

        # default power of light: light.energy * 1 = 60 W
        light = bpy.data.lights["Light"]
        light.energy = light.energy * 1

        # making sure structured light is turned off in the beginning
        self.structured_light = light.node_tree
        self.structured_light.nodes["Emission"].inputs["Strength"].default_value = 0

    # include as functions:
    def position(self, frame, x=None, y=None, z=None, pitch=None, yaw=None):
        # position at a given keyframe

        armature = bpy.context.scene.objects["Armature"]
        iris_motion_controller = armature.pose.bones["driver"]

        if type(pitch) != type(None):
            # move pitch of blender object
            # with input commands in Iris coordinate system
            # any conversion necessary to Blender coordinate system
            iris_motion_controller.rotation_euler[0] = math.radians(pitch) # important to convert to radians for Blender!
            self.pitch = pitch # update internal variables 

        if type(yaw) != type(None):
            # move yaw of blender object
            iris_motion_controller.rotation_euler[1] = math.radians(yaw)
            self.yaw = yaw

        if type(z) != type(None):
            # convert Iris coordinate system to Blender coordinate system
            # iris_z=0.0 maps to blender_y=-1.5, and iris_z=1.5 maps to blender_y=0.0
            blender_y = (z - 1.5) * 1000 # mm

            # move z of blender object
            iris_motion_controller.location[1] = blender_y # note, Blender's y equals Iris's z, so we just swap the control here, to keep the API the same as real robot
            
            # save internal coordinates in Iris coordinates 
            self.z = z

        if type(y) != type(None):
            # convert Iris coordinate system to Blender coordinate system
            # iris_y=-1.3 maps to blender_x=1.3, and iris_y=1.2 maps to blender_x=-1.2
            blender_x = -y * 1000 # mm

            # move y of blender object	 
            iris_motion_controller.location[0] = blender_x # note, Blender's x equals Iris's y...
            self.y = y

        if type(x) != type(None):
            # convert Iris coordinate system to Blender coordinate system
            # iris_x=-1.3 maps to blender_z=-1.3, and iris_x=1.2 maps to blender_z=1.2
            blender_z = x * 1000 # mm

            # move x of blender object
            iris_motion_controller.location[2] = blender_z # note, Blender's z equals Iris's x...
            self.x = x
        
        # keyframing the position 
        iris_motion_controller.keyframe_insert(data_path='location', frame=frame)
        iris_motion_controller.keyframe_insert(data_path='rotation_euler', frame=frame)


    def scan(self, frame, number_of_frames_to_scan=30):
        # start scanning at "frame" (e.g. frame = 0) for a defined number_of_frames_to_scan
        # ensure that keyframes are dedicated for showing structured light projections .mov

        emission = self.structured_light.nodes["Emission"]
        offset = self.structured_light.nodes["Image Texture"].image_user
        
        # Set to True if fps hack is used to debug
        debug_mode = True
        if debug_mode:
            adjusted_value = number_of_frames_to_scan
        else:
            adjusted_value = 1
        
        adjacent_frame = 1 / adjusted_value
        
        # ek #
        # emission keyframes (ek): measure to disable smooth transition
        emission.inputs["Strength"].default_value = 0
        emission.inputs["Strength"].keyframe_insert(data_path='default_value', frame = frame - adjacent_frame)
        #light is now OFF
        emission.inputs["Strength"].default_value = 1
        emission.inputs["Strength"].keyframe_insert(data_path='default_value', frame=frame)
        #light is now ON
        # ek #

        offset.frame_offset = 1
        offset.keyframe_insert(data_path='frame_offset', frame = frame)
        
        offset.frame_offset = 170
        offset.keyframe_insert(data_path='frame_offset', frame = frame + number_of_frames_to_scan)

        # ek #
        emission.inputs["Strength"].default_value = 1
        emission.inputs["Strength"].keyframe_insert(data_path='default_value', frame=frame+number_of_frames_to_scan)
        #light is now ON
        emission.inputs["Strength"].default_value = 0
        emission.inputs["Strength"].keyframe_insert(data_path='default_value', frame=frame + number_of_frames_to_scan + adjacent_frame)
        #light is now OFF
        # ek #

        # to ensure machine doesn't move during scanning, we can just set the machine position after scanning to be the same as its starting position
        iris.position(frame = frame + number_of_frames_to_scan, x=self.x, y=self.y, z=self.z, pitch=self.pitch, yaw=self.yaw) 


    def add_new_object_to_scan(self, filepath="monstera_with_pot.glb", delete_existing=False, rescale_ratio=1.0, initial_x=0.0, initial_y=0.0, initial_z=0.0):
        # put an "object of interest" inside Iris to be 3D scanned by the machine

        # delete existing 3D model, if needed
        if delete_existing:
            # example code for deleting everything; actually, not good here; best thing would to identify model object by name, then use delete command below
            for o in bpy.context.scene.objects:
                o.select_set(True)
                bpy.ops.object.delete()
       
        # get existing objects  
        existing_objects = set(bpy.context.scene.objects)
       
        # import .glb that has been previously cleaned up and formatted
        ext = filepath.split(".")[-1]
        if ext == ".glb" or ".gltf":
            bpy.ops.import_scene.gltf(filepath=filepath)

        imported_object = list(set(bpy.context.scene.objects) - existing_objects)[0]
        print("Import objects with names: {}".format(imported_object))

        # rescale imported object
        imported_object.delta_scale = (rescale_ratio, rescale_ratio, rescale_ratio)
        # imported_object.location.x = initial_x * 1000
        # imported_object.location.y = initial_y * 1000
        # imported_object.location.z = initial_z * 1000
        imported_object.location = bpy.context.scene.cursor.location ## cursor is exactly in the middle of iris' floor


    def open_doors(self, start_frame, end_frame):
        # ||------------------------------------------||
        # ||          ||    CLOSED    ||      OPEN    ||
        # || DOOR     || LEFT | RIGHT || LEFT | RIGHT ||
        # ||------------------------------------------||
        # || FULLY    ||  180 | -180  || -90  |  90   ||
        # || PARALLEL ||  180 | -180  ||   0  |   0   ||
        # ||------------------------------------------||
        #            (iris top view) 
        #   fully:                parallel: 
        #                         __     __
        #  ||   ||                  |   |
        #   |___|                   |___|

        closed = 180
        open = 90 # set to 90 for fully, to 0 for parallel

        left_door = bpy.data.objects["left_door"]
        right_door = bpy.data.objects["right_door"]

        # CLOSED DOORS
        left_door.rotation_euler[2] = math.radians(closed)
        left_door.keyframe_insert(data_path='rotation_euler', frame=start_frame)

        right_door.rotation_euler[2] = math.radians(-closed)
        right_door.keyframe_insert(data_path='rotation_euler', frame=start_frame)
        # CLOSED DOORS

        # OPEN DOORS
        left_door.rotation_euler[2] = math.radians(-open)
        left_door.keyframe_insert(data_path='rotation_euler', frame=end_frame)

        right_door.rotation_euler[2] = math.radians(open)
        right_door.keyframe_insert(data_path='rotation_euler', frame=end_frame)
        # OPEN DOORS

def load_iris():
    # load iris.blend as the starting point of entire animation
    # fine to bake in settings via the GUI, and minimize Python API to only essential functions for dynamic scripting of animations
    filepath = "Iris.blend"
    iris_blend = bpy.ops.wm.open_mainfile(filepath=filepath)
    # during the loading of the scene, instantiate the Iris() object above and then return it
    iris = Iris()
    return iris

    
def blender_camera_view(frame, x=None,y=None,z=None,pitch=None,yaw=None,roll=None):
    # a simple function for controlling the position of rendering camera in Blender!
    # this function should operate with the exact same coordinate system as Iris
    # that is, repeat coordinate system mapping above, so that it becomes easy, e.g., to overlap the Blender renderer and the Iris scanner view
    # LOCAL
    # access the camera in blender 
    # camera = bpy.context.scene.camera # get 'Camera' object from scene
    camera = bpy.context.scene.objects["Camera"]
    
    # HACK: for now just set camera's coordinates by GUI, plugging in those values 
    # Eventually, would be good if same API could be used, but not vital right now

    if type(pitch) != type(None):
        # iris_pitch = 90 for blender_euler_x = 180
        # blender_euler_x = pitch + 90 
        # move camera pitch (same as Iris coordinate system)
        camera.rotation_euler[0] = math.radians(pitch) # important to convert to radians for Blender!

    if type(yaw) != type(None):
        # move camera yaw
        #camera.rotation_euler[2] = math.radians(yaw)
        camera.rotation_euler[1] = math.radians(yaw)

    if type(roll) != type(None):
        camera.rotation_euler[2] = math.radians(roll) # important to convert to radians for Blender!

    if type(z) != type(None):
        # move camera z
        camera.location[2] = z 
    if type(y) != type(None):
        # move camera y
        camera.location[1] = y 

    if type(x) != type(None):
        # convert Iris coordinate system to Blender camera coordinate system

        # move camera x
        camera.location[0] = x

    camera.keyframe_insert(data_path='location', frame=frame)
    camera.keyframe_insert(data_path='rotation_euler', frame=frame)

def fps(input_frame_value):
    # change fps based on resolution parameter (good for debugging slow frame rate)
   return int(fps_resolution * input_frame_value)

# def fps(input_frame_value):
#     # change fps based on resolution parameter (good for debugging slow frame rate)
#     # to easily switch back if needed
#    return input_frame_value

    
if __name__ == "__main__":
    # load machine components, set up motion drivers
    iris = load_iris()
    
    # load 3D model to show off getting scanned inside of Iris
    iris.add_new_object_to_scan(filepath="/home/threeco/Desktop/3cology/research/simulation/models/monstera/monstera_with_pot_centered.glb", 
                                rescale_ratio=0.01,
                                initial_x=3.8805,
                                initial_y=2.1864, 
                                initial_z=0.6169)

    # camera view is defined in the same coordinate system as the Iris scanner (even if may be outside)
    # in this case, we are setting the camera to be outside of the doors of the scanner, facing inside, through the doors
    blender_camera_view(frame=fps(0), x=3.7326*1000, y=10.125*1000, z=1.5858*1000, pitch=90, yaw=0.0, roll=180)
    
    # initialize position of Iris in center, folded up, facing down (a good safe place to start any scan)
    iris.position(frame=fps(0), x=1.5, y=1.5, z=0.75, pitch=0.0, yaw=45.0)

    # start with a scan from the top
    iris.scan(frame=fps(0), number_of_frames_to_scan=fps(10))
    
    # initialize position of Iris in center, folded up, facing down (a good safe place to start any scan)
    iris.position(frame=fps(20), x=1.0, y=1.5, z=1.00, pitch=0.0, yaw=90.0)


    # insert keyframes for opening doors here
    # iris.open_doors(start_frame, end_frame)

    # # for a user camera facing into the machine, through the doors from the outside, we then see machine move away from camera, 
    # # to the opposite side of the doors, to the position x=1.5 meters, and to the right, to y=1.5 meters, over the next 30 frames
    # iris.position(frame=fps(30), x=1.5, y=1.5)


