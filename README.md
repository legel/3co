### 3D scanning simulator in Blender + Python

*3D models with geometry, diffuse, normals, and roughness rendered in simulator:*
![](https://github.com/3cology/research/blob/master/simulation/outputs/full_res_chair.png)
![](https://github.com/3cology/research/blob/master/simulation/outputs/full_res_tire.png)
![](https://github.com/3cology/research/blob/master/simulation/outputs/full_res_pillow.png)

Instructions for installing and developing on the simulator, with optics and photonics modeled after the Iris 3D scanning system by 3co.

#### Install via Command Line Terminal
0. Get this directory on your computer  
   `git clone https://github.com/3cology/research.git`  
   `cd research`

1. Download Blender LTS Release 2.83.13 [here](https://www.blender.org/download/lts/ "here"). Unzip.

2. Add Blender to command line path ([instructions for Linux, Mac, Windows](https://docs.blender.org/manual/en/2.79/render/workflows/command_line.html "instructions")).  
   For Mac:  
   ```echo "alias blender=/Applications/Blender.app/Contents/MacOS/Blender" >> ~/.bash_profile```  
   For Ubuntu:  
   ```echo "alias blender=/home/3co/blender-2.83.13-linux64/blender" >> ~/.bashrc```  
3. `source ~/.bash_profile` (Mac) or `source ~/.bashrc` (Linux)
4. Run Blender command to get path of its Python installation:  
   `blender -b -P check_python_executable_path.py`
5. Copy and paste into terminal the output line that includes "blender_py".  
   For Mac:  
   ```echo "alias blender_py=/Applications/Blender.app/Contents/Resources/2.82/python/bin/python3.7m" >> ~/.bash_profile```  
   For Ubuntu:   
   ```echo "alias blender_py=/home/3co/blender-2.83.13-linux64/2.83/python/bin/python3.7m" >> ~/.bashrc```  
6. `source ~/.bash_profile` (Mac) or `source ~/.bashrc` (Linux)
7. Prepare to install new modules into this Python:  
   ```blender_py -m ensurepip```  
   ```blender_py -m pip install --upgrade pip setuptools wheel```
8. Here's how to install any missing modules, including these that will be needed:  
   ```blender_py -m pip install Pillow```  
   ```blender_py -m pip install opencv-python```  
   ```blender_py -m pip install imageio```  

#### Let there be renders
The simulator is based in simulation/simulator.py, with an example to get one view implemented in the main function: 

```python
if __name__ == "__main__": 
  iris = Iris(model="/pillow/pillow.glb", resolution=0.1)
  iris.view(x=0, y=1.15, z=1.2, rotation_x=45, rotation_y=0.0, rotation_z=180)
  iris.scan(exposure_time=0.025, scan_id=1)
```

In path_planning.py, there's have different paths for the pillow: 
```
path = path_planning.get_pillow_path_small()    # to get a few
path = path_planning.get_pillow_path()          # really big dataset

iris = Iris(model="/pillow/pillow.glb", resolution=1.0)
startOfflineSimulation(iris=iris, exposure_time=0.015, path=path)
```

Decide whether to run the code on gpu or cpu by setting `device=` to either. Then you can run the above via:
  `blender --python simulator.py -- device=cpu`
 
If you want to add a render configuration for the Principled BSDF shader:
`blender --python simulator.py -- device=cpu render_config=render_config.json`

This `render_config.json` file should use the names that blender uses. For example:
```
{
"Base Color" : [0.23, 0.87, 0.48, 1.0],
"Metallic" : 0.1,
"Subsurface": 0.2,
"Specular": 0.3,
"Roughness": 0.4,
"Specular Tint": 0.5,
"Anisotropic": 0.6,
"Sheen": 0.7,
"Sheen Tint": 0.8,
"Clearcoat": 0.9,
"Clearcoat Roughness" : 1.0
}
```

To use the code in the cloud, this will work nicely:  
  `DISPLAY=:0 blender --python simulator.py -- device=gpu`

This opens a dummy display for the GUI to virtually show up in. If you want to run multiple simulations at the same time, or for some reason a display doesn't work, set up a new one as following:
  
Make sure you are in the `research/simulation` directory, which is where the dummy display configuration lives.

To set up a display N, run 
`sudo X :N -config dummy-1920x1080.conf`
To run:
`DISPLAY=:N blender --python simulator.py -- device=gpu`
So for example, a new display 1:
`sudo X :1 -config dummy-1920x1080.conf`
`DISPLAY=:1 blender --python simulator.py -- device=gpu`
