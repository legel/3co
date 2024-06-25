from time import time, ctime

def print_with_time(statement="", debug=True):
	current_time = ctime(time())
	if debug:
		print("[{}] {}".format(current_time, statement))