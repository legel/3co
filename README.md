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
   For Mac: `echo "alias blender=/Applications/Blender.app/Contents/MacOS/Blender" >> ~/.bash_profile`  
   For Ubuntu: `echo "alias blender=/home/3co/blender-2.83.13-linux64/blender" >> ~/.bashrc`  
3. `source ~/.bash_profile` or `source ~/.bashrc`
4. Run Blender command to get path of its Python installation:  
   `blender -b -P check_python_executable_path.py`
5. Copy and paste into terminal the output line that includes "blender_py".  
   For Mac: `blender_py=/Applications/Blender.app/Contents/Resources/2.81/python/bin/python3.7m`  
   For Ubuntu: `echo "alias blender_py=/path/to/blender-2.82a-linux64/2.82/python/bin/python3.7m" >> ~/.bash_profile`  
6. `source ~/.bash_profile`
5. Prepare to install new modules into this Python:  
   `blender_py -m ensurepip`
6. Here's how to install any missing modules, including these that will be needed:  
   `blender_py -m pip install Pillow`

#### Let there be renders
The simulator is based in simulation/simulator.py, with an example implemented in the main function: 

```python
if __name__ == "__main__": 
  iris = Iris(model="/pillow/pillow.glb", resolution=0.1)
  iris.view(x=0, y=1.15, z=1.2, rotation_x=45, rotation_y=0.0, rotation_z=180)
  iris.scan(exposure_time=0.025, scan_id=1)
```
You can run the above via:  
  `blender --python simulator.py -- cpu`

In the cloud, this will work nicely too:  
  `blender -noaudio -b --python simulator.py -- gpu`
