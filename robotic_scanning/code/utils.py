import sys
import os

def quietly_execute(f):
    old_stdout = start_ignore_prints()
    outputs = f()
    end_ignore_prints(old_stdout)
    return outputs

def start_ignore_prints():
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")
    return old_stdout

def end_ignore_prints(old_stdout):
    sys.stdout = old_stdout