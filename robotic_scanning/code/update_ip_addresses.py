import socket
import requests
from this_computer import this_computer_name
import time
import json
import os

while True:
    # get latest ip address of this computer
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80)) 
        ip_address = s.getsockname()[0]
        print(ip_address)
        time.sleep(5.0)
    except OSError as e:
        print('Unable to reach: {}'.format(e))
        continue

    # update latest ip address of this computer
    try:
        r = requests.post('http://35.175.183.124/update-ip-address?computer={}&ip_address={}'.format(this_computer_name, ip_address))
        computers = json.loads(r.content.decode('utf-8'))
        print(computers)
    except requests.exceptions.ConnectionError:
        print('Error connecting to our 3co IP server')
        time.sleep(10.0)
        continue

    # save latest ip addresses of all computers
    with open('{}/3cobot/ip_addresses.py'.format(os.getenv('HOME')), 'w') as ip_addresses:
        ip_addresses.write('ip_addresses = {')
        for computer in computers:
            ip_addresses.write('\'{}\': \'{}\', '.format(computer, computers[computer]))
        ip_addresses.write('}\n')

    # rinse and repeat
    time.sleep(10.0)
