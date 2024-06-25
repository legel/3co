import subprocess

def photograph(file_name = "", first_photo_in_sequence = True, iso = "100", exposure_comp = "0.0", aperture = "22.0", shutter_speed = "1.4", resolution = "24M", quality = "3", focus = "-f", focus_all = True):
    if first_photo_in_sequence:
        settings = "--select_af_point=Auto-5 --exposure_compensation={} --iso={} --aperture={} --shutter_speed={} --resolution={} --quality={} --jpeg_image_tone=Auto {} --white_balance_mode=Auto".format(exposure_comp, iso, aperture, shutter_speed, resolution, quality, focus)
    else:
        if focus_all:
            settings = "-f"
    command = "./pktriggercord-cli {} -o {}.jpg".format(settings, file_name).replace("  ", " ")
    print('\nLaunching command:\n{}'.format(command))
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, cwd="/home/sense/pktriggercord")
    output, error = process.communicate()
    return [output, error]

if __name__ == '__main__':
    photograph(file_name = 'x4', first_photo_in_sequence = False)
