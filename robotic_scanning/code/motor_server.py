from calibrations import *
from this_computer import *
from tkinter import *
import Slush
import gui
import RPi.GPIO as GPIO
import random
from time import sleep
from ultrasonic_sensor import MySensor
from commander import not_busy
import socket
import sys
import time

# launch a socket server on this computer, listening for messages containing commands
socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
socket_server.bind(('0.0.0.0', 8080))
socket_server.listen(5)

# initialize Slush Engine drivers for stepper motors
board = Slush.sBoard()
app = gui.Window()

engines = {}


class Engine:
    def __init__(self, axis):
        self.axis = axis
        self.port = motors[axis]['port']
        microsteps = motors[axis]['microsteps']

        self.motor = Slush.Motor(int(self.port))
        self.motor.setMicroSteps(int(microsteps))
        self.motor.setAsHome()
        self.coordinate = None
        
        max_speed = motors[self.axis].get('max_speed', None)
        accel = motors[self.axis].get('accel', None)
        decel = motors[self.axis].get('decel', None)

        hold_current = motors[self.axis].get('hold_current', None)
        run_current = motors[self.axis].get('run_current', None)
        acc_current = motors[self.axis].get('acc_current', None)
        dec_current = motors[self.axis].get('dec_current', None)

        if type(hold_current) != type(None) and type(run_current) != type(None) and type(acc_current) != type(None) and type(dec_current) != type(None):
            self.motor.setCurrent(hold_current, run_current, acc_current, dec_current)
        
        if type(max_speed) != type(None):
            self.motor.setMaxSpeed(int(max_speed))

        if type(accel) != type(None):
            self.motor.setMaxSpeed(int(accel))
            
        if type(decel) != type(None):
            self.motor.setMaxSpeed(int(decel))
        
        self.motor.setLowSpeedOpt(1)
        
        #if self.axis == 'z1' or self.axis == 'z2':
        #    self.position_initialized = True
        #else:
        #    self.position_initialized = False

        self.position_initialized = False
            
        if motors[self.axis]['positioning'] == 'live_sensor':
            pass # initialization occurs simultaneously for multiple axes during boot-up
        elif motors[self.axis]['positioning'] == 'recorded_motion':
            with open('{}.txt'.format(self.axis), 'r') as position:
                self.coordinate = float(position.readline().rstrip("\n").replace(" ", ""))
        elif motors[self.axis]['positioning'] == 'manual_initialization':
            if motors[axis]['motion'] == 'angular':
                self.coordinate = int(motors[self.axis]['steps_from_initialization_to_origin']) / float(motors[self.axis]['steps_per_degree'])
            elif motors[axis]['motion'] == 'linear':
                self.coordinate = 0.0 # initialize at 0 for wherever position is
        print('Motor {} initialized at {} (degrees or meters from origin)'.format(axis, self.coordinate))


    #goUntilPress(ACT,DIR,SPD)  ACT=0,resets ABS_POS reg, DIR = 0 or 1
    def calibrate(self):
        if self.axis == 'camera_focus':
            self.move(-10)
            self.recoordinate()
            app.update_position(self.axis, 0.0)
            return

        if self.axis == 'camera_aperture':
            self.move(-20)
            self.recoordinate()
            app.update_position(self.axis, 0.0)
            return

        if self.axis == 'projector_focus':
            self.move(-10)
            self.recoordinate()
            app.update_position(self.axis, 0.0)
            return

        if motors[self.axis]['motion'] == 'linear':
            steps_per_unit = motors[self.axis]['steps_per_mm']
            unit = 'mm'
        elif motors[self.axis]['motion'] == 'angular':
            steps_per_unit = motors[self.axis]['steps_per_degree']
            unit = 'degrees'

        directional_correction = int(motors[self.axis]['directional_correction'])
        if directional_correction == -1:
            direction_1 = 0
            direction_2 = 1
        else:
            direction_1 = 1
            direction_2 = 0

        if (self.axis == "z1"): calibration_speed = 7500
        elif(self.axis == "z2"): calibration_speed = 7500
        elif(self.axis == "phi"): calibration_speed = 1000
        elif(self.axis == "y"): calibration_speed = 10000
        elif(self.axis == "theta"): calibration_speed = 1000
        else: calibration_speed = 10000
        
        if (self.axis == "x1" or self.axis == "x2"):
            x1_engine = get_engine('x1')
            x2_engine = get_engine('x2')
            
            x1_engine.motor.goUntilPress(0, 1, 20000)
            x2_engine.motor.goUntilPress(0, 0, 20000)
            
            while x1_engine.motor.isBusy() or x2_engine.motor.isBusy():
                pos_x1 = x1_engine.motor.getPosition()
                pos_x2 = x2_engine.motor.getPosition()
                if pos_x1 != 0:
                    endPosition_x1 = pos_x1
                if pos_x2 != 0:
                    endPosition_x2 = pos_x2
                app.update_position(x1_engine.axis, pos_x1)
                app.update_position(x2_engine.axis, pos_x2)
                continue
            endPosition = (endPosition_x1+endPosition_x2)/2

            x1_engine.motor.move(-1000)
            x2_engine.motor.move(1000)

            while x1_engine.motor.isBusy() or x2_engine.motor.isBusy():
                continue

            x1_engine.motor.goUntilPress(0, 1, 100)
            x2_engine.motor.goUntilPress(0, 0, 100)

            while x1_engine.motor.isBusy() or x2_engine.motor.isBusy():
                continue

            x1_engine.motor.move(-1000)
            x2_engine.motor.move(1000)

            while x1_engine.motor.isBusy() or x2_engine.motor.isBusy():
                continue
            x1_engine.recoordinate()
            x2_engine.recoordinate()
            x1_engine.coordinate = 0.0
            x2_engine.coordinate = 0.0
            pos_x1 = x1_engine.motor.getPosition()
            pos_x2 = x2_engine.motor.getPosition()
            app.update_position(x1_engine.axis, pos_x1)
            app.update_position(x2_engine.axis, pos_x2)

            self.recoordinate()
            self.coordinate = 0.0
            pos = self.motor.getPosition()
            app.update_position(self.axis, pos)
        
            print ("distance traveled: " + str(endPosition) + " steps")
        
            print ("Length of travel: " + str(endPosition/steps_per_unit) + " " + str(unit))


        elif (self.axis == "theta"):
            
            initial_position = self.motor.getPosition()
            print("Initial position of: {}".format(initial_position))
            
			# start by moving into one limit switch
            self.motor.goUntilPress(0, direction_1, calibration_speed)
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    end_position = pos
                app.update_position(self.axis, pos)
                continue
                
			#print(end_position)
            
            #end_position += directional_correction*700               
            # back up from a limit switch
            self.motor.move(-1*directional_correction*700)
            while self.motor.isBusy():
                continue
                
			# approach first limit switch slowly
            self.motor.goUntilPress(0, direction_1, int(calibration_speed/10)) 
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    end_difference = pos
                app.update_position(self.axis, pos)
                continue
                
            #end_position += end_difference
            
            #final_position = self.motor.getPosition()
            #print("Final position of: {}".format(end_position))

            self.recoordinate()
            
            self.move(-173.5)
            while self.motor.isBusy():
                continue
            self.recoordinate()

        elif (self.axis == "phi"):
            
            initial_position = self.motor.getPosition()
            print("Initial position of: {}".format(initial_position))
            
            # start by moving into one limit switch
            self.motor.goUntilPress(0, direction_1, calibration_speed)
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    end_position = pos
                app.update_position(self.axis, pos)
                continue
                
            #print(end_position)
            #self.recoordinate()

            #end_position += directional_correction*700               
            # back up from a limit switch
            self.motor.move(-directional_correction*2000)
            while self.motor.isBusy():
                continue
                
            # approach first limit switch slowly
            self.motor.goUntilPress(0, direction_1, int(calibration_speed/10)) 
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    end_difference = pos
                app.update_position(self.axis, pos)
                continue
               
            self.recoordinate()
            #end_position += end_difference
            
            #final_position = self.motor.getPosition()
            #print("Final position of: {}".format(end_position))
            

            self.move(-180)
            while self.motor.isBusy():
                continue
            self.recoordinate()

            self.move(-12.0)
            while self.motor.isBusy():
                continue
            self.recoordinate()


			## move to the 2nd limit switch          
            #self.motor.goUntilPress(0, direction_2, calibration_speed)
            #while self.motor.isBusy():
                #continue
		        
            #self.motor.move(directional_correction*200)
            #while self.motor.isBusy():
                #continue
                
            #self.motor.goUntilPress(0, direction_2, int(calibration_speed/20))
            #while self.motor.isBusy():
                #continue
		        
            #limit_switch_two_position = self.motor.getPosition()
            
            #print("Current position is: {}".format(limit_switch_two_position))
            
            #steps_from_limit_switch_two_to_one = limit_switch_two_position
            #self.motor.goTo(int(steps_from_limit_switch_two_to_one/2.0)) 
            #while self.motor.isBusy():
                #continue
            #self.recoordinate()
            #app.update_position(self.axis, 0.0)
		
        elif self.axis == "z1":
            # now we calibrate z by moving up, so we flip the directions
            directional_correction = int(motors[self.axis]['directional_correction'])
            if directional_correction == -1:
                direction_1 = 1
                direction_2 = 0
            else:
                direction_1 = 0
                direction_2 = 1
            
            # move the motor to the first limit switch
            self.motor.goUntilPress(0, direction_2, calibration_speed)

            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    endPosition = pos
                app.update_position(self.axis, pos)
                continue
            

	    # back up from a limit switch
            self.motor.move(-1*directional_correction*300)
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                app.update_position(self.axis, pos)
                continue

	    # approach first limit switch slowly
            self.motor.goUntilPress(0, direction_2, int(calibration_speed/100)) 
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    temp = pos
                app.update_position(self.axis, pos)
                continue
                
            # back up again off the limit switch
            self.motor.move(-1*directional_correction*2000)
                
            endPosition = abs(endPosition + temp)
            print (str(endPosition)+"  steps & travel: " + str(endPosition/steps_per_unit) + " " + str(unit))
            sys.stdout.flush()
            
            #self.recoordinate()	
            self.coordinate = 0.85
            self.move(0.85)
            pos = self.motor.getPosition()
            app.update_position(self.axis, pos)	

        elif self.axis == "z2":
            # now we calibrate z by moving up, so we flip the directions
            directional_correction = int(motors[self.axis]['directional_correction'])
            if directional_correction == -1:
                direction_1 = 1
                direction_2 = 0
            else:
                direction_1 = 0
                direction_2 = 1
                
            # move the motor to the first limit switch
            self.motor.goUntilPress(0, direction_2, calibration_speed)

            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    endPosition = pos
                app.update_position(self.axis, pos)
                continue
            

	    # back up from a limit switch
            self.motor.move(-1*directional_correction*300)
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                app.update_position(self.axis, pos)
                continue

	    # approach first limit switch slowly
            self.motor.goUntilPress(0, direction_2, int(calibration_speed/100)) 
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    temp = pos
                app.update_position(self.axis, pos)
                continue
                
            endPosition = abs(endPosition + temp)
            print (str(endPosition)+"  steps & travel: " + str(endPosition/steps_per_unit) + " " + str(unit))
            sys.stdout.flush()
            
            # back up again off the limit switch
            self.motor.move(-1*directional_correction*2000)
            
            #self.recoordinate()	
            self.coordinate = 0.85
            self.move(0.85)
            pos = self.motor.getPosition()
            app.update_position(self.axis, pos)	
        elif self.axis == "y":
	        # move the motor to the first limit switch
            self.motor.goUntilPress(0, direction_2, calibration_speed)

            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    endPosition = pos
                app.update_position(self.axis, pos)
                continue
            

	    # back up from a limit switch
            self.motor.move(directional_correction*300)
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                app.update_position(self.axis, pos)
                continue

	    # approach first limit switch slowly
            self.motor.goUntilPress(0, direction_2, int(calibration_speed/100)) 
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    temp = pos
                app.update_position(self.axis, pos)
                continue
                
            endPosition = abs(endPosition + temp)
            print (str(endPosition)+"  steps & travel: " + str(endPosition/steps_per_unit) + " " + str(unit))
            sys.stdout.flush()
            
            # back up again off the limit switch
            self.motor.move(directional_correction*100000)
            
            self.recoordinate()	
            self.coordinate = 0.0
            pos = self.motor.getPosition()
            app.update_position(self.axis, pos)	
            
#            self.move(1.0)
#            while self.motor.isBusy():
#                pos = self.motor.getPosition()
#                if pos != 0:
#                    temp = pos
#                app.update_position(self.axis, pos)
#                continue
#            self.recoordinate()	
#            self.coordinate = 0.0
#            pos = self.motor.getPosition()
#            app.update_position(self.axis, pos)	
           

        else:

	        # move the motor to the first limit switch
#            if self.axis == "z2":
#                self.motor.goUntilPress(0, direction_1, calibration_speed)
#            else:
            self.motor.goUntilPress(0, direction_1, calibration_speed)

            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    endPosition = pos
                app.update_position(self.axis, pos)
                continue

	    # back up from a limit switch
            self.motor.move(-directional_correction*1000)
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                app.update_position(self.axis, pos)
                continue

	    # approach first limit switch slowly
            self.motor.goUntilPress(0, direction_1, int(calibration_speed/100)) 
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    temp = pos
                app.update_position(self.axis, pos)
                continue
                
            endPosition = abs(endPosition + temp)
            print (str(endPosition)+"  steps & travel: " + str(endPosition/steps_per_unit) + " " + str(unit))
            sys.stdout.flush()
                
	    #move to the next limit switch          
            self.motor.goUntilPress(0, direction_2, calibration_speed)
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    endPosition = pos
                app.update_position(self.axis, pos)
                continue
               
            time.sleep(1.0)
            if self.axis == "z2": 
                self.motor.move(directional_correction*500)
                time.sleep(1.0)
                self.motor.move(directional_correction*500)
            else:
                self.motor.move(directional_correction*1000)

            while self.motor.isBusy():
                pos = self.motor.getPosition()
                continue
                
            self.motor.goUntilPress(0, direction_2, int(calibration_speed/100))
            while self.motor.isBusy():
                pos = self.motor.getPosition()
                if pos != 0:
                    temp = pos
                app.update_position(self.axis, pos)
                continue
		        
            endPosition = abs(endPosition + temp)
                
            print (str(endPosition)+"  steps & travel: " + str(endPosition/steps_per_unit) + " " + str(unit))
            print(str(self.axis) + ' current coordinate: '+ str(self.coordinate))

            sys.stdout.flush()
            
            #move to the zero position
            if self.axis == 'phi':
                self.motor.move(int(endPosition/2))
                while self.motor.isBusy():
                    pos = self.motor.getPosition()
                    app.update_position(self.axis, pos)
                    continue
            elif self.axis == 'z1' or self.axis == 'z2':
                self.motor.move(directional_correction*1000)
                while self.motor.isBusy():
                    continue
            else:
                self.motor.move(int(endPosition/2))
                while self.motor.isBusy():
                    continue
                    
            self.recoordinate()
            self.coordinate = 0.0
            pos = self.motor.getPosition()
            app.update_position(self.axis, pos)
        
            print ("distance traveled: " + str(endPosition) + " steps")
        
            print ("Length of travel: " + str(endPosition/steps_per_unit) + " " + str(unit))

        self.position_initialized = True
        sys.stdout.flush()                


    def move(self, position):
        #if self.position_initialized == False:
        #    self.calibrate()
        print('moving: {}'.format(self.axis))
        print('{}'.format(motors[self.axis]['motion']))
        print('from: ' + str(self.coordinate) + ' to: ' + str(position))
  
        # account for direction of positive/negative directional differences from wiring
        directional_correction = int(motors[self.axis]['directional_correction'])

        # protect against overshooting
        if position < motors[self.axis]['min']:
            print('Warning: Command to move {} to {}m is outside of safe boundary. Moving to {}m instead.'.format(self.axis, position, motors[self.axis]['min']))
            position = float(motors[self.axis]['min'])
        elif position > motors[self.axis]['max']:
            print('Warning: Command to move {} to {}m  is outside of safe boundary. Moving to {}m instead.'.format(self.axis, position, motors[self.axis]['max']))
            position = float(motors[self.axis]['max'])

        # move linearly or angularly, depending on axis
        if motors[self.axis]['motion'] == 'linear':
            steps_per_mm = motors[self.axis]['steps_per_mm']
            mm_to_position = 1000 * (float(position) - self.coordinate)  # 1000 to convert mm to meters; negative to go in opposite direction from coordinate
            stepper_position_to_move_to = steps_per_mm * mm_to_position
            print(str(self.axis)+' moving from: ' + str(self.coordinate) + ' to: ' + str(position) + ' in steps: ' + str(stepper_position_to_move_to)) 
            print('{} moving {} mm, i.e. {} steps'.format(self.axis, mm_to_position, stepper_position_to_move_to))
            stepper_position_to_move_to = stepper_position_to_move_to * directional_correction # only change actual motor command for direction
            self.motor.move(int(stepper_position_to_move_to))  
            self.coordinate = round(position, 4)
        elif motors[self.axis]['motion'] == 'angular':
            steps_per_degree = motors[self.axis]['steps_per_degree']
            degrees_to_position = position - self.coordinate
            stepper_position_to_move_to = steps_per_degree * degrees_to_position 
            print('{} moving {} degrees, i.e. {} steps'.format(self.axis, degrees_to_position, stepper_position_to_move_to))
            stepper_position_to_move_to = stepper_position_to_move_to * directional_correction # only change actual motor command for direction
            self.motor.move(int(stepper_position_to_move_to))  
            self.coordinate = round(position, 4)
       
        # save motion in a text file for the axis if needed
        if motors[self.axis]['positioning'] == 'recorded_motion':
            with open('{}.txt'.format(self.axis), 'w') as position:
                print('Saving recorded motion of {} in {}.txt as {}'.format(self.axis, self.axis, self.coordinate))
                position.write("{}".format(self.coordinate))

        sys.stdout.flush()
 
    def motor_to_position(self, position): # for testing and development uses only, use move() above for applications
        print("Attempting to move {} to {}".format(self.axis, position))
        self.motor.goTo(int(position))          

    def set_metric_position(self, current_mm_from_sensor):
        origin_mm_from_sensor = motors_to_sensors[self.axis]['origin_mm_from_sensor']
        current_position_in_meters = (current_mm_from_sensor - origin_mm_from_sensor) / 1000.0
        self.coordinate = current_position_in_meters * int(motors_to_sensors[self.axis]['orientation_of_sensor_relative_to_axis'])
        print('{} axis currently at {} meters from origin'.format(self.axis, self.coordinate))

    def stop(self):
        self.motor.hardStop()

    def recoordinate(self):
        self.coordinate = 0.0
        self.motor.setAsHome()
        if motors[self.axis]['positioning'] == 'recorded_motion':
            with open('{}.txt'.format(self.axis), 'w') as position:
                print('Recoordinated {} in {}.txt as {}'.format(self.axis, self.axis, self.coordinate))
                position.write("{}".format(self.coordinate))
        sys.stdout.flush()
        
    def get_position():
        return self.motor.getPosition()

    def free(self):
        self.motor.free()

    def is_busy(self):
        return self.motor.isBusy()

    def go_to_limit_switch(self, limit_switch_index, direction):
        speed = motors[self.axis]['max_speed']
        self.motor.goUntilPress(int(limit_switch_index), int(direction), int(speed))

def initialize_positions_from_sensors(motors_to_initialize = 'all', n_seconds = 10):
    # initialize position of motor and save result after n seconds; motors_to_initialize can be a list of motors e.g. ['x1','x2'] or left blank to do all
    if this_computer_name == 'pi_1': 
        sensors = {}
        if motors_to_initialize == 'all': # x1, x2, y
            motors_to_initialize = motors_to_sensors.keys() # imported from calibrations.py file
        for motor in motors_to_initialize:
            engine = get_engine(motor) 
            sensor = MySensor(motor, motors_to_sensors[motor]['sensor_address'])
            sensor.start()
            sensors[motor] = sensor
        sleep(n_seconds)    
        for motor in motors_to_initialize:
            sensor = sensors[motor]
            current_mm_from_sensor = sensor.average_distance()
            print('{} axis detected {} mm from sensor'.format(motor, current_mm_from_sensor))
            engine = get_engine(motor)
            engine.set_metric_position(current_mm_from_sensor)
            sensor.stop()
    #elif this_computer_name == 'pi_2':
    #    with open('positions.txt', 'r'):
            


def get_engine(axis):
    engine = engines.get(axis, None)
    if type(engine) != type(None):
        #pos = engine.motor.getPosition()
        #app.update_position(engine, pos)
        return engine
    else:
        engine = Engine(axis)
        engines[axis] = engine
        #pos = engine.motor.getPosition()
        #app.update_position(engine, pos)
        return engine

def parse(signal):
    #signal = str(signal.payload).lstrip('b').strip('\'')
    parameters = {}
    if '?' in signal:
        command = signal.split('?')[0]
        for parameter in signal.split('?')[1].split('&'):
            key, value = parameter.split('=')
            parameters[key] = value
            print('{}  =  {}'.format(key, value))
    else:
      command = signal
    return command, parameters

#def on_connect(client, userdata, flags, rc):
#    client.subscribe(network_channel_name) 

#def on_log(client, userdata, level, buff):
#    if 'Caught exception in' in buff:
#        print("\nERROR:\n{}".format(buff.split("on_message: ")[1]))

def on_message(msg):
    from calibrations import motors
    command, parameters = parse(msg)

    reply_from_server = "Your message ({}) was heard, loud and clear".format(msg)

    if command == 'move': 
        x = parameters.get('x')
        y = parameters.get('y')
        z = parameters.get('z')
        phi = parameters.get('phi')
        theta = parameters.get('theta')
        turn = parameters.get('turn')
        projector_focus = parameters.get('projector_focus')
        camera_focus = parameters.get('camera_focus')
        camera_aperture = parameters.get('camera_aperture')
        camera_polarization = parameters.get('camera_polarization')
        cali_x = parameters.get('cali_x')
        cali_y = parameters.get('cali_y')
        cali_z = parameters.get('cali_z')
        cali_turn = parameters.get('cali_turn')

        if z != 'None': # special processing for z motors, to prevent dual activity (amp overload)
            # account for z axis inverse coordinate system
            z1_max = float(motors['z1']['max'])
            z2_max = float(motors['z2']['max'])
            z1_min = float(motors['z1']['min'])
            z2_min = float(motors['z2']['min'])

            combined_max = z1_max + z2_max           
            #new_coordinate = combined_max - float(z)
            new_coordinate = float(z)
            z1_engine = get_engine('z1')
            z2_engine = get_engine('z2')
            current_coordinate = z1_engine.coordinate + z2_engine.coordinate 
        
            if current_coordinate < z1_max and new_coordinate < z1_max:
                # z1 will already be 0, and it stays 0, z2 will move to new_coordinate
                z1_engine.move(new_coordinate)

            elif current_coordinate < z1_max and new_coordinate >= z1_max:
                # z1 will move to new_coordinate - z2_max, and z2 will move to z2_max
                z1_engine.move(z1_max)
                z2_engine.move(new_coordinate - z1_max)

            elif current_coordinate >= z1_max and new_coordinate < z1_max:
                # z1 will move to 0, and z2 will move to new_coordinate
                z2_engine.move(z2_min)
                z1_engine.move(new_coordinate)

            elif current_coordinate >= z1_max and new_coordinate >= z1_max:
                # z1 will move to new_coordinate - z2_max, and z2 will stay at z2_max
                z2_engine.move(new_coordinate - z1_max)

            while z1_engine.is_busy() or z2_engine.is_busy():
                for engine in [z1_engine, z2_engine]:
                    pos = engine.motor.getPosition()
                    #pos = engine.coordinate
                    app.update_position(engine.axis, pos)
                    #print('position:' + str(pos))
                sleep(0.10)


        for axis, position in [('x',x),('y',y),('phi',phi),('theta',theta),('turn',turn),('projector_focus',projector_focus),('camera_focus',camera_focus),('camera_aperture',camera_aperture),('camera_polarization',camera_polarization),('cali_x',cali_x),('cali_y',cali_y),('cali_z',cali_z),('cali_turn',cali_turn)]:
            if position == 'None':
                continue

            # first send all commands
            if axis in multi_motor_axes.keys():
                multi_motors = multi_motor_axes[axis]
                for motor in multi_motors:
                    engine = get_engine(motor)
                    engine.move(float(position))
            else:
                engine = get_engine(axis)
                engine.move(float(position))



        for axis, position in [('x',x),('y',y),('phi',phi),('theta',theta),('turn',turn),('projector_focus',projector_focus),('camera_focus',camera_focus),('camera_aperture',camera_aperture),('camera_polarization',camera_polarization),('cali_x',cali_x),('cali_y',cali_y),('cali_z',cali_z),('cali_turn',cali_turn)]:
            if position == 'None':
                continue

            # now wait for all of them to be finished

            if axis in multi_motor_axes.keys():
                multi_motors = multi_motor_axes[axis]
                for motor in multi_motors:
                    engine = get_engine(motor)
                    while engine.is_busy():
                        pos = engine.motor.getPosition()
                        #pos = engine.coordinate
                        app.update_position(engine.axis, pos)
                        #print('position:' + str(pos))
                        sleep(0.10)
            else:
                engine = get_engine(axis)

                while engine.is_busy():
                    pos = engine.motor.getPosition()
                    #pos = engine.coordinate
                    app.update_position(engine.axis, pos)
                    #print('position:' + str(pos))
                    sleep(0.10)
        sys.stdout.flush()

    elif command == 'calibrate':
        axis = parameters.get('axis')
        if axis in multi_motor_axes.keys():
            motors = multi_motor_axes[axis]
            for motor in motors:
                engine = get_engine(motor)
                engine.calibrate()
            for motor in motors:
                engine = get_engine(motor)
                while engine.is_busy():
                    pos = engine.motor.getPosition()
                    app.update_position(engine.axis, pos)
                    sleep(0.10)
        else:
            engine = get_engine(axis)
            engine.calibrate()
            while engine.is_busy():
                pos = engine.motor.getPosition()
                app.update_position(engine.axis, pos)
                sleep(0.10)



    elif command == 'motor_to_position': 
        axis = parameters.get('axis')
        position = parameters.get('position')
        if axis in multi_motor_axes.keys():
            motors = multi_motor_axes[axis]
            for motor in motors:
                engine = get_engine(motor)
                engine.motor_to_position(position) # run command on motor

    elif command == 'recoordinate':
        axis = parameters.get('axis') 
        if axis in multi_motor_axes.keys():
            motors = multi_motor_axes[axis]
            for motor in motors:
                engine = get_engine(motor)
                engine.recoordinate()
        else:
            engine = get_engine(axis)
            engine.recoordinate()

    elif command == 'stop':
        axis = parameters.get('axis') 
        if axis in multi_motor_axes.keys():
            motors = multi_motor_axes[axis]
            for motor in motors:
                engine = get_engine(motor)
                engine.stop()
        else:
            engine = get_engine(axis)
            engine.stop()


    elif command == 'free': 
        axis = parameters.get('axis') 
        if axis in multi_motor_axes.keys():
            motors = multi_motor_axes[axis]
            for motor in motors:
                engine = get_engine(motor)
                engine.free()
        else:
            engine = get_engine(axis)
            engine.free()


#    elif command == 'go_to_limit_switch':	 
#        axis = parameters.get('axis') 
#        limit_switch_index = parameters.get('limit_switch_index') 
#        direction = parameters.get('direction') 
#        engine = get_engine(axis)
#        engine.go_to_limit_switch(limit_switch_index, direction)

    elif command == 'die':
        GPIO.cleanup()

    elif command == 'signal_when_finished':
        axes = parameters.get('axes')
        if "," in axes:
            axes = axes.split(",")
        else:
            axes = [axes]
        axes_processed_on_this_computer = []
        for axis in axes:
            if axis_to_computer[axis] == this_computer_name: # if this axis is processed on this computer
                axes_processed_on_this_computer.append(axis)
                print('Waiting for {} on {}'.format(axis, this_computer_name))
                if axis in multi_motor_axes.keys():
                    motors = multi_motor_axes[axis]
                    for motor in motors:
                        engine = get_engine(motor)
                        while engine.is_busy():
                            sleep(0.1) # sleep until each motor is not busy
                else:
                    engine = get_engine(motor)
                    while engine.is_busy():
                        sleep(0.1) # sleep until each motor is not busy
        reply_from_server = (',').join(axes_processed_on_this_computer)
        #not_busy(axes_processed_on_this_computer) # broadcast update to other computers

    return reply_from_server

if __name__ == "__main__": 
    try:

        
        #initialize_positions_from_sensors(motors_to_initialize = 'all', n_seconds = sensor_calibration_time)
        while True:
            socket_connection, _ = socket_server.accept()
            message_from_commander = ''
            while True:
                bytes_data = socket_connection.recv(4096) # bytes to receive, per loop
                if not bytes_data: break
                message_from_commander += bytes_data.decode()
                print(message_from_commander)
                reply_from_server = on_message(message_from_commander)
                socket_connection.send(reply_from_server.encode())        
                
            socket_connection.close()
    finally:
      GPIO.cleanup()
