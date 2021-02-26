import math



def rotate(x, y, z, yaw, pitch, theta):

  a = 1.0*math.radians(theta)
  xp = x*math.cos(a) - y*math.sin(a)
  yp = x*math.sin(a) + y*math.cos(a)
  zp = z
  yaw = yaw - theta
  if yaw >= 360.0:
    yaw = yaw - 360.0
  elif yaw < 0:
    yaw = yaw + 360.0

  return [xp, yp, zp, yaw, pitch]


def rotateMotion(theta, path, fps, sec):

  p = path[-1]
  for i in range(1, int(fps*sec)+1):
    p = rotate(p[0], p[1], p[2], p[3], p[4], theta/(fps*sec))
    path.append(p)

def zMotion(z, path, fps, sec):
  p = path[-1]
  for i in range(1, int(fps*sec)+1):
    q = p.copy()
    q[2] = q[2] + z/(fps*sec)
    path.append(q)
    p = q

def phiMotion(phi, path, fps, sec):
  p = path[-1]
  for i in range(1, int(fps*sec)+1):
    q = p.copy()
    q[4] = q[4] + phi/(fps*sec)
    path.append(q)
    p = q

def xyMotion(x, y, path, fps, sec):
  p = path[-1]
  for i in range(1, int(fps*sec)+1):
    q = p.copy()
    q[0] = q[0] + x/(fps*sec)
    q[1] = q[1] + y/(fps*sec)
    path.append(q)
    p = q


def get_theta_test_path():
  fps = 30.0
  path = []
  p = [0.85, 0.0, -1.2, 270.0, 0.0]
  path.append(p)
  
  for i in range(0, 360):
    p = path[-1]
    theta = p[3] + 1
    if (theta >= 360):
      theta = theta - 360
    q = p.copy()
    q[3] = theta
    path.append(q)
  
  return path


def get_chalice_path():
  fps = 1.0
  path = []
  p = [4.8, 0.0, 1.7, 90.0, 90.0]
  path.append(p)
  for i in range(12):
    rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)

  return path


def get_brownchair_path():
  fps = 1.0
  path = []

  # chair
  #p = [1.75, 0.0, 1.2, 90.0, 65.0]

  
  p = [1.75, 0.0, 2.0, 90.0, 45.0]
  path.append(p)
  for i in range(12):
    rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)

  zMotion(z=-1.5, path=path, fps=fps, sec=1.0)
  phiMotion(phi=45.0, path=path, fps=fps, sec=1.0)

  for i in range(12):
    rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  

  zMotion(z=-2.0, path=path, fps=fps, sec=1.0)
  phiMotion(phi=45.0, path=path, fps=fps, sec=1.0)

  for i in range(12):
    rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)



  return path



def get_bplant_path():
  fps = 1.0
  path = []

  # chair
  #p = [1.75, 0.0, 1.2, 90.0, 65.0]

  
  p = [1.0, 0.0, 1.6, 90.0, 35.0]
  path.append(p)
  for i in range(12):
    rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)

  zMotion(z=-1.6, path=path, fps=fps, sec=1.0)
  xyMotion(x=0.5, y=0.0, path=path, fps=fps, sec=1.0)
  phiMotion(phi=55.0, path=path, fps=fps, sec=1.0)

  for i in range(12):
    rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)


  return path

  phiMotion(phi=45.0, path=path, fps=fps, sec=1.0)


  for i in range(12):
    rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)


  # chair
  #zMotion(z=-1.55, path=path, fps=fps, sec=1.0)
  # bplant
  zMotion(z=-1.3, path=path, fps=fps, sec=1.0)
  xyMotion(x=-0.9, y=0.0, path=path, fps=fps, sec=1.0)

  # chair
  #phiMotion(phi=60.0, path=path, fps=fps, sec=1.0)
  # bplant
  phiMotion(phi=80.0, path=path, fps=fps, sec=1.0)

  for i in range(12):
    rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)


  return path


def get_circular_path():

  fps = 1.0
  path = []

  # chair
  p = [1.3, 0.0, 0.35, 90.0, 75.0]

  # bplant
  #p = [1.25, 0.0, 0.9, 90.0, 75.0]

  path.append(p)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  rotateMotion(theta=30.0, path=path, fps=fps, sec=1.0)
  return path

def get_mock_path():


  fps = 30.0
  path = []
  # start
  p = [0.85, 0.0, -1.5, 270.0, 0.0]

  path.append(p)

  # rotate 315 degrees
  rotateMotion(theta=-315.0, path=path, fps=fps, sec=3.0)
  
  # go up 0.67
  zMotion(z=0.67, path=path, fps=fps, sec=1.0)

  # rotate 22 degrees
  rotateMotion(theta=-22.0, path=path, fps=fps, sec=0.5)

  # look down
  phiMotion(phi=60.0, path=path, fps=fps, sec=0.5)

  # look forward
  phiMotion(phi=-60.0, path=path, fps=fps, sec=0.5)

  # go up 0.5
  zMotion(z=0.5, path=path, fps=fps, sec=0.9)

  # rotate 68 degrees
  rotateMotion(theta=-68.0, path=path, fps=fps, sec=1.0)

  # look down
  phiMotion(phi=60.0, path=path, fps=fps, sec=0.5)

  # go up 0.2
  zMotion(z=0.2, path=path, fps=fps, sec=0.45)

  # rotate 90 degrees
  rotateMotion(theta=-90.0, path=path, fps=fps, sec=1.2)

  # go forward 
  xyMotion(x=0.2, y=0.2, path=path, fps=fps, sec=0.5)

  # look forward
  phiMotion(phi=-60.0, path=path, fps=fps, sec=0.5)

  # go down 0.3
  zMotion(z=-0.3, path=path, fps=fps, sec=0.5)

  # look down
  phiMotion(phi=40.0, path=path, fps=fps, sec=0.5)

  # go back 
  xyMotion(x=-0.2, y=-0.2, path=path, fps=fps, sec=0.8)

  # look forward
  phiMotion(phi=-40.0, path=path, fps=fps, sec=0.5)

  # rotate 360
  rotateMotion(theta=-360.0, path=path, fps=fps, sec=3.45)

  return path


def path_to_animation_code(path):

  f_out = open("path_code.txt", "w")
  for i in range(0, len(path)):
    p = path[i]
    xd = p[0]
    yd = p[1]
    zd = p[2]
    yawd = 1.0 * (p[3])
    pitd = 1.0 * (p[4])
    f_out.write("{} {} {} {} {}\n".format(xd,yd,zd,yawd,pitd))
    print("{} {} {} {} {}\n".format(xd,yd,zd,yawd,pitd))

if __name__ == "__main__":
  #path = get_mock_path()
  #path_to_animation_code(path)
  path = get_chalice_path()
  for i in range(0, len(path)):
    print (path[i])


