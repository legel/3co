target_distance = 0.720
maximum_in_focus_distance = 1.390
focal_length = 0.012
f_stop = 2.8 # current guess, contact jeremy@ajile.ca to confirm exact value

circle_of_confusion = abs(maximum_in_focus_distance - target_distance) * focal_length**2 / (maximum_in_focus_distance * f_stop * (target_distance - focal_length) )

print("Circle of Confusion ~ {} mm".format(1000*circle_of_confusion))