import socket
from calibrations import *
from this_computer import *
from ip_addresses import ip_addresses

def convert_pitch_yaw_to_phi_theta(axes_inputs):
    if type(axes_inputs) == type({'a':'b'}):
        if type(axes_inputs.get("pitch", None)) != type(None):
            axes_inputs["phi"] = axes_inputs["pitch"]
            del axes_inputs["pitch"]
        if type(axes_inputs.get("yaw", None)) != type(None):
            axes_inputs["theta"] = axes_inputs["yaw"]
            del axes_inputs["yaw"]
    elif type(axes_inputs) == type(['a', 'b']):
        if 'pitch' in axes_inputs:
            axes_inputs.remove('pitch')
            axes_inputs.append('phi')
        if 'yaw' in axes_inputs:
            axes_inputs.remove('yaw')
            axes_inputs.append('theta')
    elif type(axes_inputs) == type('a'):
        if axes_inputs == 'pitch':
            axes_inputs = 'phi'
        elif axes_inputs == 'yaw':
            axes_inputs = 'theta'
    return axes_inputs

def move(directions, vertical_before_horizontal=True):
    # moves axis to position in either meters (linear x,y,z) or degrees (angular phi, theta), relative to origin that is center of system  e.g. move({'x': 0.5, 'y': 0.5, 'z': 0.5, 'phi': 180, 'theta': 90}
    directions = convert_pitch_yaw_to_phi_theta(directions)

    commands_for_computers = {}

    x = directions.get('x', 'None')
    y = directions.get('y', 'None')
    z = directions.get('z', 'None')
    phi = directions.get('phi', 'None')
    theta = directions.get('theta', 'None')
    turn = directions.get('turn', 'None')
    projector_focus = directions.get('projector_focus', 'None')
    camera_focus = directions.get('camera_focus', 'None')
    camera_aperture = directions.get('camera_aperture', 'None')

    directions = {'x': x, 'y': y, 'z': z, 'phi': phi, 'theta': theta, 'turn': turn, 'projector_focus': projector_focus, 'camera_focus': camera_focus, 'camera_aperture': camera_aperture}

    for computer in computers:
        if computer in axis_to_computer.values():
            spots = {'x': 'None', 'y': 'None', 'z': 'None', 'phi': 'None', 'theta': 'None', 'turn': 'None', 'projector_focus': 'None', 'camera_focus': 'None', 'camera_aperture': 'None'}
            for axis in ['x', 'y', 'z', 'phi', 'theta', 'turn', 'projector_focus', 'camera_focus', 'camera_aperture' ]:
                computer_for_this_axis = axis_to_computer[axis]
                if computer_for_this_axis == computer:
                    spots[axis] = directions[axis]
            commands_for_computers[computer] = 'move?x={}&y={}&z={}&phi={}&theta={}&turn={}&projector_focus={}&camera_focus={}&camera_aperture={}'.format(spots['x'], spots['y'], spots['z'], spots['phi'], spots['theta'], spots['turn'], spots['projector_focus'], spots['camera_focus'], spots['camera_aperture'])
 
    send(commands_for_computers, vertical_before_horizontal)

def signal_when_finished(axes = motors.keys()):
    axes = convert_pitch_yaw_to_phi_theta(axes)

    # requests a signal when motors finished for axes, e.g. signal_when_finished(['x1', 'x2']) or signal_when_finished('y')
    commands_for_computers = {computer: [] for computer in computers if computer in axis_to_computer.values()}
    if type(axes) == type(''):
      axes = [axes]
    axes = (',').join(axes)
    for computer in computers:
        if computer in axis_to_computer.values():
            command = 'signal_when_finished?axes={}'.format(axes)
            commands_for_computers[computer].append(command)
    send(commands_for_computers)
        
def motor_to_position(directions):
    # moves individual motors to specific position in relative stepper motor coordinates - direct control over motors -  e.g. motor_to_position({'x1': 100000, 'x2': 100000, 'z1': 50000, 'z2': 10000, 'phi1': 10000, 'phi2': -10000, 'y': 0, 'theta': 0}
    directions = convert_pitch_yaw_to_phi_theta(directions)

    commands_for_computers = {computer: [] for computer in computers}
    for axis in directions.keys(): 
        command = 'motor_to_position?axis={}&position={}'.format(axis, directions[axis])
        computer = axis_to_computer[axis]
        commands_for_computers[computer].append(command)
    send(commands_for_computers)

def stop(axes = motors.keys()):
    axes = convert_pitch_yaw_to_phi_theta(axes)

    # hard stop motors, e.g. stop(['x1', 'x2']) or stop('y')
    commands_for_computers = {computer: [] for computer in computers}
    if type(axes) == type(''):
      axes = [axes]
    for axis in axes: 
        command = 'stop?axis={}'.format(axis)
        computer = axis_to_computer[axis]
        commands_for_computers[computer].append(command)
    send(commands_for_computers)

def free(axes = motors.keys()):
    axes = convert_pitch_yaw_to_phi_theta(axes)
    # free connection to motors, e.g. free(['x1', 'x2']) or free('y')
    commands_for_computers = {computer: [] for computer in computers}
    if type(axes) == type(''):
      axes = [axes]
    for axis in axes: 
        command = 'free?axis={}'.format(axis)
        computer = axis_to_computer[axis]
        commands_for_computers[computer].append(command)
    send(commands_for_computers)

def recoordinate(axes = motors.keys()):
    axes = convert_pitch_yaw_to_phi_theta(axes)
    # sets current motor position to origin, i.e. 0, in stepper coordinates, e.g. recoordinate(['x1', 'x2']) or recoordinate('y')
    commands_for_computers = {computer: [] for computer in computers}
    if type(axes) == type(''):
      axes = [axes]
    for axis in axes: 
        command = 'recoordinate?axis={}'.format(axis)
        computer = axis_to_computer[axis]
        commands_for_computers[computer].append(command)
    send(commands_for_computers)

def calibrate(axes = motors.keys()):
    axes = convert_pitch_yaw_to_phi_theta(axes)
    # calibrate axis, e.g. calibrate(['x1', 'x2']) or calibrate('y')
    commands_for_computers = {computer: [] for computer in computers}
    if type(axes) == type(''):
      axes = [axes]
    for axis in axes: 
        command = 'calibrate?axis={}'.format(axis)
        computer = axis_to_computer[axis]
        commands_for_computers[computer].append(command)
    send(commands_for_computers)

def not_busy(axes):
    axes = convert_pitch_yaw_to_phi_theta(axes)
    # communicates with all other computers to update motor status as not busy, e.g. not_busy(['x', 'y'])
    commands_for_computers = {computer: [] for computer in computers}
    if type(axes) == type(''):
      axes = [axes]
    axes = (',').join(axes)
    for computer in computers:
        command = 'not_busy?axes={}'.format(axes)
        commands_for_computers[computer].append(command)
    send(commands_for_computers)

def busy(axis): 
    axis = convert_pitch_yaw_to_phi_theta(axis)
    # mark motor as busy until broadcast received to change this
    with open('motion_status/{}.py'.format(axis), 'w') as output_file:
        output_file.write('busy = True')

def send(commands_for_computers, vertical_before_horizontal=True):
    if vertical_before_horizontal:
        computers_to_command = ['pi_2', 'pi_1']
    else:
        computers_to_command = ['pi_1', 'pi_2'] 

    for computer in computers_to_command:
        commands = commands_for_computers[computer]
        if type(commands) == type(''):
            commands = [commands]
        if len(commands) > 0:
            for command in commands:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect((ip_addresses[computer], 8080))
                client.send(command.encode())
                response_from_server = client.recv(4096)
                #print(response_from_server.decode())
                client.close()
