import paho.mqtt.client as mqtt 

channel_name = 'busy_server' # a separate server to run in parallel to main one, which avoids channel communication clashes

def parse(signal):
    signal = str(signal.payload).lstrip('b').strip('\'')
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

def on_connect(client, userdata, flags, rc):
    client.subscribe(channel_name) 

def on_log(client, userdata, level, buff):
    if 'Caught exception in' in buff:
        print("\nERROR:\n{}".format(buff.split("on_message: ")[1]))

def on_message(client, userdata, msg):
    command, parameters = parse(msg)

    if command == 'not_busy':
        axes = parameters.get('axes')
        if "," in axes:
            axes = axes.split(",")
        else:
            axes = [axes]
        for axis in axes:
            print('{} not busy'.format(axis))
            with open('motion_status/{}.py'.format(axis), 'w') as output_file:
                output_file.write('busy = False')
        print(' ')

if __name__ == "__main__": 
   client = mqtt.Client()
   client.on_connect = on_connect
   client.on_message = on_message
   client.on_log = on_log
   client.connect('localhost', 1883, 60)
   client.loop_forever()

