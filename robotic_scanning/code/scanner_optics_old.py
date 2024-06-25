import time
import numpy as np
import matplotlib.pyplot as plt
import optimizer.pyqt_fit.nonparam_regression as smooth
from optimizer.pyqt_fit import npr_methods

polarization_lenses = True

if not polarization_lenses:
	# without polarization lenses, these values work
	distances = 		[275.174, 306.539, 336.916, 367.396, 407.918, 438.157, 488.493, 538.693, 613.766, 688.410, 788.626, 887.188, 1032.60, 1176.94, 1445.80, 1586.95, 1731.00]
	camera_focuses = 	[136.200, 182.000, 184.500, 193.700, 204.500, 214.400, 224.500, 230.900, 238.400, 246.100, 253.100, 261.400, 267.200, 271.900, 282.200, 283.400, 289.20 ]
	projector_focuses = [15.080,  17.300,  18.000,  18.700,  20.210,  20.650,  22.210,  22.580,  23.690,  24.070,  26.050,  26.560,  26.810,  27.070,  27.660,  27.830,  28.310 ]
	exposure_times = 	[9.200,   9.600,   10.800,  12.900,  16.600,  17.300,  22.000,  24.700,  32.900,  40.600,  41.900,  43.200,  44.500,  45.800,  47.100,  48.400,  49.700 ]
else:
	# with polarization lenses
	# distances = 		[275.174, 306.539, 336.916, 367.396, 407.918, 438.157, 488.493, 538.693, 613.766, 688.410, 788.626, 887.188, 1032.60, 1176.94, 1445.80, 1586.95, 1731.00]
	# camera_focuses = 	[146.812, 182.000, 184.500, 193.700, 204.500, 214.400, 224.500, 230.900, 238.400, 246.100, 253.100, 261.400, 267.200, 271.900, 282.200, 283.400, 289.20 ]
	# projector_focuses = [20.664,  17.300,  18.000,  18.700,  20.210,  20.650,  22.210,  22.580,  23.690,  24.070,  26.050,  26.560,  26.810,  27.070,  27.660,  27.830,  28.310 ]

	# #  original data for aperture = 50 degrees (distances not based on center point)
	# distances =  		[302.0224,				322.52524, 			342.82205, 			383.49582, 			464.93814, 			626.95404,			979.3754,			1679.9701]
	# camera_focuses = 	[140.46983928774725,	149.14416413907026,	156.7064940828933,	168.16799258936476,	186.51981075968578,	207.86840184306865,	228.22620710919688,	245.81179984225204]
	# projector_focuses = [16.87103261551947,		18.095875777884874,	19.106724440294812,	20.459299129269382,	22.69581156400338,	25.551284197079717,	27.34510327804505,	29.589174521374872]
	# exposure_times = 	[1.0600000000000005,	1.2600000000000007,	1.4600000000000009,	1.9600000000000013,	3.2599999999999967,	8.159999999999982,	18.710000000000132,	54.30999999999905]

	# data for 75.0 degrees aperture, extracted from log file
	distances =  			[195.0, 		341.945831,	361.776886,	381.738800,	421.888458,	502.716736,	665.006653,	985.424744,	1189.634888]
  # projector_distances = 	[356.326630,	377.347382,	398.371735,	440.333496,	523.887817,	689.587891,	1013.38934,	1218.796143]
	camera_focuses = 		[72.0,			149.38,		150.98,		161.69,		172.73,		188.87,		210.04,		224.38,		228.64]
	projector_focuses = 	[14.0,			23.56,		24.21,		24.89,		26.40,		29.18,		31.20,		34.05,		35.33]
	exposure_times = 		[10.0,			14.0,		18.5,		20.5,		24.5,		34.5,		55.0,		117.5,		179.13]


	# distances = []
	# camera_focuses = []
	# projector_focuses = []
	# exposure_times = [57.0, ]

	# # from aperture_50.0degrees_optics.tsv
	# distances = 		[300.0126,320.21857,340.54895,380.59735,461.0548,621.2911,935.2955,1536.1979]
	# camera_focuses = 	[136.52318662628628,144.82178293204748,151.8036799152459,165.23526161080002,183.66956874066196,206.52246343020926,227.6845731726814,243.5839149473988]
	# projector_focuses = [15.619105738293397,16.488271583120493,19.34215239895408,20.02726577284288,21.805867040922685,24.632672847594474,26.95529490922368,28.597905255119645]
	# exposure_times = 	[3.209999999999997,3.9099999999999944,4.5599999999999925,6.009999999999987,8.509999999999987,14.660000000000075,32.86000000000027,92.75999999999686]

	# from aperture_70.0degrees_optics.tsv
	# distances = 		[298.66037,				319.25986,			339.54623,			380.30328,			460.62814,			621.1854,			933.54236,			1598.4906]
	# camera_focuses = 	[133.2528669071734,		142.16679378097504,	146.53087431898444,	159.337324777352,	179.36386813087273,	203.7542222060419,	222.7331804123421,	240.4755968290543]
	# projector_focuses = [16.158885145067714,	17.03364839380303,	18.259718075999093,	19.715098391544693,	21.863953041444724,	24.27591030560532,	26.831158291529782,	28.264096875913427]
	# exposure_times = 	[12.51,					12.01,				13.01,				15.01,				21.259999999999998,	38.76,				83.25999999999999,	238.26]

	#min_exposure_times = [0.250,   9.600,   10.800,  12.900,  16.600,  17.300,  22.000,  24.700,  32.900,  40.600,  41.900,  43.200,  44.500,  45.800,  47.100,  48.400,  49.700 ]
	#max_exposure_times = [0.250,   9.600,   10.800,  12.900,  16.600,  17.300,  22.000,  24.700,  32.900,  40.600,  41.900,  43.200,  44.500,  45.800,  47.100,  48.400,  49.700 ]

exposure_times = [exposure_time * 1.0 for exposure_time in exposure_times] # hack 
projector_f_stop = 2.8 # current guess, contact jeremy@ajile.ca to confirm exact value
projector_circle_of_confusion = 0.0350 # mm
projector_focal_length = 12.00 # mm

class ScannerOptics():
	def __init__(self):
		self.fit_camera_focus_to_distance()
		self.fit_projector_focus_to_distance()
		self.fit_exposure_time_to_distance()

	def fit_camera_focus_to_distance(self):
		self.camera_focus_kernel = smooth.NonParamRegression(distances, camera_focuses, method=npr_methods.LocalPolynomialKernel(q=2))
		self.camera_focus_kernel.fit()

	def fit_projector_focus_to_distance(self):
		self.projector_focus_kernel = smooth.NonParamRegression(distances, projector_focuses, method=npr_methods.LocalPolynomialKernel(q=2))
		self.projector_focus_kernel.fit()

	def fit_exposure_time_to_distance(self):
		self.exposure_time_kernel = smooth.NonParamRegression(distances, exposure_times, method=npr_methods.LocalPolynomialKernel(q=2))
		self.exposure_time_kernel.fit()

	def plot_camera_focus_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. camera focus")
		plt.plot(distances, camera_focuses, 'o', alpha=0.5)
		plt.plot(grid, self.camera_focus_kernel(grid), 'y', linewidth=2)
		plt.ylabel("camera focus ring rotation (degrees)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("camera_focuses", "distances"))
		print("Camera angle (degrees) evaluated at 300mm, 400mm, 500mm: {}".format(self.camera_focus_kernel([300.0, 400.0, 500.0])))

	def plot_projector_focus_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. projector focus")
		plt.plot(distances, projector_focuses, 'o', alpha=0.5)
		plt.plot(grid, self.projector_focus_kernel(grid), 'y', linewidth=2)
		plt.ylabel("projector focus ring rotation (degrees)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("projector_focuses", "distances"))
		print("Projector angle (degrees) evaluated at 300mm, 400mm, 500mm: {}".format(self.projector_focus_kernel([300.0, 400.0, 500.0])))

	def plot_exposure_time_to_distance(self):
		grid = np.r_[min(distances):max(distances):1731.0j]
		plt.clf()
		plt.title("target distance vs. exposure times")
		plt.plot(distances, exposure_times, 'o', alpha=0.5)
		plt.plot(grid, self.exposure_time_kernel(grid), 'y', linewidth=2)
		plt.ylabel("exposure time (milliseconds)")
		plt.xlabel("distance (millimeters)")
		plt.savefig("{}_vs_{}.png".format("exposure_times", "distances"))
		print("Exposure time (ms) evaluated at 300mm, 400mm, 500mm: {}".format(self.exposure_time_kernel([300.0, 400.0, 500.0])))

	def camera_focus(self, distance):
		return self.camera_focus_kernel([distance])[0]

	def projector_focus(self, distance):
		return self.projector_focus_kernel([distance])[0]

	def exposure_time(self, distance):
		return self.exposure_time_kernel([distance])[0]

	def focus(self, distance):
		camera_focus = self.camera_focus(distance = distance)
		projector_focus = self.projector_focus(distance = distance)
		exposure_time = self.exposure_time(distance = distance)
		return [camera_focus, projector_focus, exposure_time]

	def depth_of_field(self, distance):
		if distance:
			minimum_in_focus_depth = distance * projector_focal_length**2 / (projector_focal_length**2 + projector_f_stop * projector_circle_of_confusion * distance)
			if distance > 1000.0:
				maximum_in_focus_depth = 3000.0
			else:
				maximum_in_focus_depth = distance * projector_focal_length**2 / (projector_focal_length**2 - projector_f_stop * projector_circle_of_confusion * distance)
		else:
			minimum_in_focus_depth = 0.0
			maximum_in_focus_depth = 10000.0
		return minimum_in_focus_depth, maximum_in_focus_depth


if __name__ == "__main__":
	start_time = time.time()
	target_distance = 1000

	optics = ScannerOptics()
	for distance in range(200, 1500, 50):
		minimum_in_focus_depth, maximum_in_focus_depth = optics.depth_of_field(distance=distance)
		camera_focus = optics.camera_focus(distance = distance)
		projector_focus = optics.projector_focus(distance = distance)
		exposure_time = optics.exposure_time(distance = distance)
		print("Targeting {:.1f}mm away, minimum in-focus depth = {:.1f}mm, maximum in-focus depth = {:.1f}mm, camera focus at {} degrees, projector focus at {} degrees, exposure time at {} seconds".format(distance, minimum_in_focus_depth, maximum_in_focus_depth, camera_focus, projector_focus, exposure_time))
	optics.plot_exposure_time_to_distance()
	optics.plot_projector_focus_to_distance()
	optics.plot_camera_focus_to_distance()

# Targeting 200.0mm away, minimum in-focus depth = 176.0mm, maximum in-focus depth = 231.5mm
# Targeting 250.0mm away, minimum in-focus depth = 213.6mm, maximum in-focus depth = 301.3mm
# Targeting 300.0mm away, minimum in-focus depth = 249.1mm, maximum in-focus depth = 377.0mm
# Targeting 350.0mm away, minimum in-focus depth = 282.7mm, maximum in-focus depth = 459.4mm
# Targeting 400.0mm away, minimum in-focus depth = 314.4mm, maximum in-focus depth = 549.6mm
# Targeting 450.0mm away, minimum in-focus depth = 344.5mm, maximum in-focus depth = 648.6mm
# Targeting 500.0mm away, minimum in-focus depth = 373.1mm, maximum in-focus depth = 757.9mm
# Targeting 550.0mm away, minimum in-focus depth = 400.2mm, maximum in-focus depth = 879.0mm
# Targeting 600.0mm away, minimum in-focus depth = 426.0mm, maximum in-focus depth = 1014.1mm
# Targeting 650.0mm away, minimum in-focus depth = 450.6mm, maximum in-focus depth = 1165.6mm
# Targeting 700.0mm away, minimum in-focus depth = 474.1mm, maximum in-focus depth = 1336.9mm
# Targeting 750.0mm away, minimum in-focus depth = 496.6mm, maximum in-focus depth = 1531.9mm
# Targeting 800.0mm away, minimum in-focus depth = 518.0mm, maximum in-focus depth = 1756.1mm
# Targeting 850.0mm away, minimum in-focus depth = 538.5mm, maximum in-focus depth = 2016.5mm
# Targeting 900.0mm away, minimum in-focus depth = 558.1mm, maximum in-focus depth = 2322.6mm
# Targeting 950.0mm away, minimum in-focus depth = 577.0mm, maximum in-focus depth = 2687.6mm
# Targeting 1000.0mm away, minimum in-focus depth = 595.0mm, maximum in-focus depth = 3130.4mm
# Targeting 1050.0mm away, minimum in-focus depth = 612.4mm, maximum in-focus depth = 3678.8mm
# Targeting 1100.0mm away, minimum in-focus depth = 629.1mm, maximum in-focus depth = 4375.7mm
# Targeting 1150.0mm away, minimum in-focus depth = 645.1mm, maximum in-focus depth = 5290.7mm
# Targeting 1200.0mm away, minimum in-focus depth = 660.6mm, maximum in-focus depth = 6545.5mm
# Targeting 1250.0mm away, minimum in-focus depth = 675.4mm, maximum in-focus depth = 8372.1mm
# Targeting 1300.0mm away, minimum in-focus depth = 689.8mm, maximum in-focus depth = 11277.1mm
# Targeting 1350.0mm away, minimum in-focus depth = 703.6mm, maximum in-focus depth = 16615.4mm
# Targeting 1400.0mm away, minimum in-focus depth = 716.9mm, maximum in-focus depth = 29647.1mm
# Targeting 1450.0mm away, minimum in-focus depth = 729.8mm, maximum in-focus depth = 109894.7mm
