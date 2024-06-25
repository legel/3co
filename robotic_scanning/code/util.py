def parse(args, defaults=None):
	for arg in args[1].split("&"):
		key, value = arg.split("=")
		defaults[key] = value
	return defaults