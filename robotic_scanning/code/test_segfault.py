import os
import signal

def sig_handler(signum, frame):
    pass
#signal.signal(signal.SIGSEGV, sig_handler)

os.kill(os.getpid(), signal.SIGSEGV)

print("And the beat goes on")

os.kill(os.getpid(), signal.SIGSEGV)

print("On and on and on")
