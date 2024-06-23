import psutil

def kill_command(command_substring_to_purge):
	pids_to_kill = []
	processes = psutil.process_iter(attrs=["pid", "cmdline"])

	for process in processes:
		pid = process.info["pid"]
		for subcommands in process.info["cmdline"]:
			#print(subcommands)
			if command_substring_to_purge in subcommands:
				#print("{} with PID {} to kill".format(subcommands, pid))
				pids_to_kill.append(pid)

	for pid in pids_to_kill:
		parent = psutil.Process(pid)
		for child in parent.children(recursive=True):  # or parent.children() for recursive=False
			child.kill()
			#print("Killing children of {}".format(pid))
		parent.kill()
		print("Killed {} of process {}".format(pid, command_substring_to_purge))


#ill_command("spherical_calibrator.py")