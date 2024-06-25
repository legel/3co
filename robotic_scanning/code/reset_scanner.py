from time import sleep

import board
import digitalio
 
power = digitalio.DigitalInOut(board.C1) # relay is connected to the C0 GPIO pin of the FT232H board
power.direction = digitalio.Direction.OUTPUT # set relay pin as output

power.value = False # Relay is active low, so when it's low, relay is open and scanner is powered OFF
print('Relay LOW, scanner powered OFF')
sleep(10) # wait for 10 seconds for the scanner to fully power OFF
power.value = True # Relay inactive, contact is closed and the scanner is powered ON
print('Relay HIGH, scanner powered ON')

