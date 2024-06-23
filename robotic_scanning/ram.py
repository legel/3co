import numpy as np
import time
import sys
import psutil


def sizeof_fmt(num, suffix='B'):
	''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
	for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
		if abs(num) < 1024.0:
			return "%3.1f %s%s" % (num, unit, suffix)
		num /= 1024.0
	return "%.1f %s%s" % (num, 'Yi', suffix)

def how_much_ram_is_being_used():
	''' Memory usage in GB '''

	def get_size(bytes, suffix="B"):
	    """
	    Scale bytes to its proper format
	    e.g:
	        1253656 => '1.20MB'
	        1253656678 => '1.17GB'
	    """
	    factor = 1024
	    for unit in ["", "K", "M", "G", "T", "P"]:
	        if bytes < factor:
	            return f"{bytes:.2f}{unit}{suffix}"
	        bytes /= factor


	with open('/proc/self/status') as f:
		memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

	svmem = psutil.virtual_memory()
	print(f"\n\nRAM Total Virtual Use: {svmem.percent}%")
	swap = psutil.swap_memory()
	print(f"RAM Total Swap Use: {swap.percent}%")

	print("\n{:.2f} GB of RAM currently being used by this program\n\n".format(int(memusage.strip()) / (1000.0 * 1000.0)), flush=True)

