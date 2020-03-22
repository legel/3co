### 3D scanning simulator in Blender + Python

*One of 100 cleaned 3D models we have for testing, rendered from 3 perspectives:*
![](https://raw.githubusercontent.com/stev3/research/master/assets/reconstructable_1.png?token=AAOTUSS7CHNRK46PM4FNW3K6P7UWI)
![](https://raw.githubusercontent.com/stev3/research/master/assets/reconstructable_2.png?token=AAOTUSRRNBPCZH5G7BCJPHK6P7U6A)
![](https://raw.githubusercontent.com/stev3/research/master/assets/reconstructable_3.png?token=AAOTUSVD2PBE4GYCA7ZVV4C6P7VAC)

Instructions for installing and developing on 3co's raycast+render-powered RGB point cloud simulator, which uses the same optics as our scanner and motion coordinate system as our robot.

#### Install via Command Line Terminal
0. Get this directory on your computer  
  `git clone https://github.com/stev3/research.git`  
  `cd research`
1. Download Blender [here](https://www.blender.org/download/ "here")
2. Add Blender to command line path ([instructions for Linux, Mac, Windows](https://docs.blender.org/manual/en/2.79/render/workflows/command_line.html "instructions")), *e.g.*  
  `echo "alias blender=/Applications/Blender.app/Contents/MacOS/Blender" >> ~/.bash_profile`  
  `source ~/.bash_profile`
3. Run Blender command to get path of its Python installation:  
  `blender -b -P check_python_executable_path.py`
4. Copy and paste into terminal the output line that includes "blender_py", *e.g.*  
  `blender_py=/Applications/Blender.app/Contents/Resources/2.81/python/bin/python3.7m`
5. Prepare to install new modules into this Python:  
`$blender_py -m ensurepip`
6. Here's how to install any missing modules, including this one for reading render images:  
`$blender_py -m pip install Pillow`
7. Download 100 cleaned 3D models for testing  
  `curl -O https://3co.s3.amazonaws.com/reconstructables.zip`
8. Unzip that directory  
  `unzip reconstructables.zip`

(Incidentally rendered images of the entire dataset can be downloaded and viewed [here](https://3co.s3.amazonaws.com/renders_360.zip "here"))

#### Let there be raycasts
The simulator is currently entirely in optics.py, with an example implemented in the main function:  

```python
if __name__ == "__main__":  
  environment = Environment()

  sensor_resolution = 0.25 # set to 1.0 for full resolution equivalent to our scanner
  sensors = Optics( photonics="sensors", 
                    environment=environment, 
                    focal_point=Point(2.0, 0.0, 0.0), 
                    focal_length=0.012, 
                    vertical_pixels=2280 * sensor_resolution, 
                    horizontal_pixels=1824 * sensor_resolution, 
                    pixel_size=0.00000587 / sensor_resolution,
                    target_point=Point(0.0,0.0,0.0))

  scanner = Scanner(sensors=sensors, environment=environment)
  models = get_3D_models()

  for i, model in enumerate(models):
    environment.new_model(model)
    outputs = scanner.scan(x=1.25, y=0.0, z=0.0, pitch=90, yaw=90, turntable=0)

    print("render: {}".format(outputs["render_file"]))
    print("(x,y,z,r,g,b) + pixel position (h,v): {}".format(outputs["point_cloud_file"]))
    print("3D model w/ mesh: {}".format(outputs["3D_model_file"]))
  
    # e.g. load outputs above, process, continue scanning and processing below... 

    outputs = scanner.scan(x=1.1, y=0.0, z=0.25, pitch=75, yaw=90, turntable=30) 
    
  # ...
```
You can just run the above in optics.py by running the following:  
  `blender -b --python optics.py -noaudio -- 0 simulated_scanner_outputs`


#####Notes:
* Colors of (x,y,z) points in .csv and .ply output file are based on real renders (i.e. shadows, effects of lighting).  We will need to be able to build a model that accounts for the perspective that a color was seen.
* Position of (x,y,z) points are based on original global values in the 3D model.  Therefore for the purpose of aligning point clouds and for measuring error, these values are valid.
* Scanner is based on our robotic coordinate system, i.e. (x,y,z,pitch,yaw,turntable). A bit of further documentation/thinking [here](https://docs.google.com/document/d/1FsgnzzdmZE0qz_1uw7lePc5e3lh1HGlXNSBlKcXP4hU/edit "here").