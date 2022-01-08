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

class Iris():
# create an Iris class that simplifies the API for working with Blender
# include as attributes:
	def __init__(self, x=0.0, y=0.0, z=1.7, pitch=-90.0, yaw=0.0):
		# position attributes
		self.x = x # interfacing with the Blender coordinate systems, define x = 0 to be equal to center of machine (if you're facing into machine from doors, set +x equal further away)
		self.y = y # ... y = 0 to be equal to center of machine in y-axis of the machine (y-axis is the one with overhanging bridge; if you're facing into the machine from doors, set +y = right side)
		self.z = z # ... z = 0 to be equal to the floor of the machine, touching the top of it (z-axis is up and down; set +z to be up)
		self.pitch = pitch # ... pitch = 0 to be facing forward, like a human head starting directly ahead (set +pitch upward as high as +90 degrees, -pitch downward as low as -90 degrees)
		self.yaw = 0 # ... yaw = 0 to be equal to facing the opening doors of the machine (set +yaw = turning to the right side from our view facing into doors, up to 179.99 degrees; -yaw turns to the left side back to -179.99 degrees)

# include as functions:
	def position(self, frame, x=None, y=None, z=None, pitch=None, yaw=None):
		# frame will define the keyframe number of the animation that Iris should be in this position
		# this way, e.g., we can issue a sequence of positions and then Blender (as it currently does) 
		# will automatically compute the motion in between each position at the speed needed

		iris_motion_controller = # get driver in blender

		if type(pitch) != type(None):
			# move pitch of blender object, with input commands in Iris coordinate system, and any conversion necessary to Blender coordinate system
			iris_motion_controller.rotation_euler[0] = math.radians(pitch) # important to convert to radians for Blender!
			self.pitch = pitch # update internal variables 

		if type(yaw) != type(None):
			# move yaw of blender object
			iris_motion_controller.rotation_euler[1] = math.radians(yaw)
			self.yaw = yaw

		if type(z) != type(None):
			# move z of blender object
			iris_motion_controller.location[2] = z # note, in Blender's coordinate system, the third parameter of location is Z; however, for whatever reason, things might be switched up, relative to Iris coordinate system; just hack and work around
			self.z = z

		if type(y) != type(None):
			# move y of blender object	 
			iris_motion_controller.location[1] = y # note, in Blender's coordinate system, the second parameter of location is Y; ...
			self.y = y

		if type(x) != type(None):
			# move x of blender object
			iris_motion_controller.location[0] = x # note, in Blender's coordinate system, the first parameter of location is X; ...
			self.x = x


	def scan(self, frame, number_of_frames_to_scan=30):
		# start scanning at "frame" (e.g. frame = 0) for a defined number_of_frames_to_scan
		# ensure that keyframes are dedicated for showing structured light projections .mov

		# [insert scanning script here]

		# to ensure machine doesn't move during scanning, we can just set the machine position after scanning to be the same as its starting position
		iris.position(frame=frame + number_of_frames_to_scan, x=self.x, y=self.y, z=self.z, pitch=self.pitch, yaw=self.yaw) 


	def add_new_object_to_scan(filepath="monstera_with_pot.glb", delete_existing=False):
		# put an "object of interest" inside Iris to be 3D scanned by the machine

		# delete existing 3D model, if needed
		if delete_existing:
			# example code for deleting everything; actually, not good here; best thing would to identify model object by name, then use delete command below
			for o in bpy.context.scene.objects:
	  			o.select_set(True)
				bpy.ops.object.delete()

		# import .glb that has been previously cleaned up and formatted
		ext = filepath.split(".")[-1]
		if ext == ".glb" or ".gltf":
			bpy.ops.import_scene.gltf(filepath=filepath)



def load_iris():
	# load iris.blend as the starting point of entire animation
	# fine to bake in settings via the GUI, and minimize Python API to only essential functions for dynamic scripting of animations

	# during the loading of the scene, instantiate the Iris() object above and then return it
	iris = Iris()
	return iris

def rescale_model_size(rescale_ratio=1.0):
	# rescale model size based on argument to import function (since different models may need different rescaling)
	# model = // get model by name
	model.delta_scale = (rescale_ratio, rescale_ratio, rescale_ratio)


def blender_render_view(x=None,y=None,z=None,pitch=None,yaw=None):
	# a simple function for controlling the position of rendering camera in Blender!
	# this function should operate with the exact same coordinate system as Iris
	# that is, repeat coordinate system mapping above, so that it becomes easy, e.g., to overlap the Blender renderer and the Iris scanner view

	# access the camera in blender 
	camera = # get 'Camera' object from scene

	if type(pitch) != type(None):
		# move camera pitch (same as Iris coordinate system)
        camera.rotation_euler[0] = math.radians(pitch) # important to convert to radians for Blender!

	if type(yaw) != type(None):
		# move camera yaw
		camera.rotation_euler[1] = math.radians(yaw)

	if type(z) != type(None):
		# move camera z
		camera.location[2] = z # note, in Blender's coordinate system, the third parameter of location is Z; however, for whatever reason, things might be switched up, relative to Iris coordinate system; just hack and work around
	if type(y) != type(None):
		# move camera y
		camera.location[1] = y # note, in Blender's coordinate system, the second parameter of location is Y; ...

	if type(x) != type(None):
		# move camera x
		camera.location[0] = x # note, in Blender's coordinate system, the first parameter of location is X; ...


if __name__ == "__main__":
	# load machine components, set up motion drivers
	iris = load_iris()

	# load 3D model to show off getting scanned inside of Iris
	iris.add_new_object_to_scan("/path/to/monstera_with_pot.glb")

	# camera view is defined in the same coordinate system as the Iris scanner (even if may be outside)
	# in this case, we are setting the camera to be outside of the doors of the scanner, facing inside, through the doors
	blender_render_view(frame=0, x=-5.0, y=0.0, z=1.7, pitch=0.0, yaw=-180.0)

	# initialize position of Iris in center, folded up, facing down (a good safe place to start any scan)
	iris.position(frame=0, x=0.0, y=0.0, z=1.7, pitch=-90.0, yaw=0.0)

	# capture a scan to start the scene, to survey what's inside
	iris.scan(frame=0, number_of_frames_to_scan=30)

	# for a user camera facing into the machine, through the doors from the outside, we then see machine move away from camera, 
	# to the opposite side of the doors, to the position x=1.5 meters, and to the right, to y=1.5 meters, over the next 30 frames
	iris.position(frame=30, x=1.5, y=1.5)

	# capture another survey scan, at the far right corner of the machine, facing down, this time only a short exposure
	iris.scan(frame=30, number_of_frames_to_scan=5)

	# machine moves along the right side for the next 60 frames (frame 35 to frame 95), coming toward us, all the way to the nearest edge, by the doors
	iris.position(frame=95, x=-1.5)

	# capture another quick scan, near right corner of the machine
	iris.scan(frame=95, number_of_frames_to_scan=5)

	# machine moves to the left side, from our perspective facing into the doors, to the position y=-1.5 meters
	iris.position(frame=160, y=-1.5)

	# capture another scan
	iris.scan(frame=160, number_of_frames_to_scan=5)

	# machine moves away from camera, to the opposite side of the doors, again, but this time along the left side 
	iris.position(frame=210, x=1.5)

	# capture another scan
	iris.scan(frame=210, number_of_frames_to_scan=5)

	# machine now re-centered in y-axis
	iris.position(frame=245, y=0.0)

	# machine moves its head from facing down to facing forward, along the direction parallel to the x-axis (going to and from the doors)
	iris.position(frame=275, pitch=0.0)

	# machine moves down from its starting 1.7 meters up position, to 0.85 meters above the floor
	iris.position(frame=335, z=0.85)

	# now, we would like to move the camera into the same view as Iris
	# to make sure the camera has been still the entire time for previous actions
	# we must set again the same camera position at the latest frame
	blender_render_view(frame=335, x=-5.0, y=0.0, z=1.7, pitch=0.0, yaw=-180.0)

	# now, let's animation the camera motion, zooming into the perspective of Iris
	blender_render_view(frame=395, x=1.5, y=0.0, z=0.85, pitch=0.0, yaw=0.0)

	# capture another scan, this time a long exposure again; for the first time, we see the scan from Iris's perspective
	iris.scan(frame=395, number_of_frames_to_scan=30)

	# machine moves back to the far right again, but this time with its z-axis extended half-way, and its head facing forward 
	iris.position(frame=455, y=1.5)

	# as well, as move the camera in synchrony with the machine! "I am become Iris, 3D modeler of worlds."
	blender_render_view(frame=455, y=1.5)

	# machine is in the far right corner, and now orients its head so that it should now be facing to the left
	iris.position(frame=485, yaw=-90)
	blender_render_view(frame=485, yaw=-90)

	# machine moves along the right side, coming toward us, stopping halfway in the center of x-axis
	iris.position(frame=515, x=0)
	blender_render_view(frame=515, x=0)

	# capture the last survey scan, one last long exposure
	iris.scan(frame=515, number_of_frames_to_scan=30)