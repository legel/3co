import argparse
import os
import time

class Replicator:
	def __init__(self):
		self.start_time = time.time()

		# high-level control parameters for each of the stages of the pipeline, useful for skipping stages (use pattern --no-[X] below, e.g. --no-preprocess to skip preprocess) 
		parser = argparse.ArgumentParser()
		parser.add_argument('--preprocess', type=bool, default=True, help='Flag to preprocess (or not) raw scan data; use flag \'--no-preprocess\' to not preprocess', action=argparse.BooleanOptionalAction)
		parser.add_argument('--naive_low_resolution_training', type=bool, default=True, help='Flag to do naive low resolution training (or not); use flag \'--no-naive_low_resolution_training\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--naive_high_resolution_training', type=bool, default=True, help='Flag to do naive high resolution training (or not); use flag \'--no-naive_high_resolution_training\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--generate_point_clouds_from_naive_model', type=bool, default=True, help='Flag to make point clouds (or not) from naive NeRF trained model; use flag \'--no-generate_point_clouds_from_naive_model\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--merge_naive_point_clouds', type=bool, default=True, help='Flag to merge point clouds or not; use flag \'--no-merge_naive_point_clouds\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--voxelize_point_clouds', type=bool, default=True, help='Flag to voxelize point clouds or not; use flag \'--no-voxelize_point_clouds\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--focused_high_resolution_training', type=bool, default=True, help='Flag to do focused high resolution training (or not); use flag \'--no-focused_high_resolution_training\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--generate_point_clouds_from_focused_model', type=bool, default=True, help='Flag to generate high resolution focused point clouds for review', action=argparse.BooleanOptionalAction)
		parser.add_argument('--merge_focused_point_clouds', type=bool, default=True, help='Flag to merge point clouds or not; use flag \'--no-merge_focused_point_clouds\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--naive_extraction_of_depth_distributions', type=bool, default=True, help='Flag to do focused extraction of depth distributions (or not); use flag \'--no-focused_extraction_of_depth_distributions\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--sdf_training', type=bool, default=True, help='Flag to do SDF training (or not); use flag \'--no-sdf_training\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--export_point_clouds_of_sdf', type=bool, default=True, help='Flag to extract point clouds from 50th percentile of SDF (or not); use flag \'--no-sdf_training\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--merge_point_clouds_of_sdf', type=bool, default=True, help='Flag to merge point clouds of pre-computed 50th percentile depths (or not); use flag \'--no-merge_point_clouds_of_sdf\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--extract_sdf_field', type=bool, default=True, help='Flag to extract SDF field (or not); use flag \'--no-extract_sdf_field\' to skip if already done', action=argparse.BooleanOptionalAction)
		parser.add_argument('--export_extrinsics_intrinsics', type=bool, default=True, help='Whether to export camera extrinsics and intrinsics (or not)', action=argparse.BooleanOptionalAction)

		# parameters for scan data
		parser.add_argument('--scan_preprocessor', type=str, default='/home/photon/sense/StrayVisualizer/convert_to_open3d.py', help='Location of the preprocessing script to run on raw scan data from iPhone')
		parser.add_argument('--scan_directory', type=str, default='/home/photon/sense/3cology/plantvine/scans/bird_of_paradise', help='Location of raw uploaded 3D scan data')
		
		# parameters across all phases of training
		parser.add_argument('--number_of_images_in_training_dataset', type=int, default=320, help='Number of images to train on')
		parser.add_argument('--number_of_samples_outward_per_raycast', type=int, default=320, help='The number of samples per raycast to collect for each rendered pixel during training')
		parser.add_argument('--number_of_samples_outward_per_raycast_for_test_renders', type=int, default=320, help='The number of samples per raycast to collect for each rendered pixel during testing')

		# parameters for "naive" (unbiased coarse sampling) initial NeRF run at low resolution, where every pixel has a depth loss value
		parser.add_argument('--naive_low_resolution_epochs', type=int, default=50001, help='Number of epochs to run NeRF in \'naive\' coarse sampling mode at low resolution, with no constraint on ray sampling, to auto-focus on zone-of-interest voxels')
		parser.add_argument('--naive_low_resolution_image_H', type=int, default=int(480), help='Training image pixel height during \'naive\' mode') 
		parser.add_argument('--naive_low_resolution_image_W', type=int, default=int(640), help='Training image pixel width during \'naive\' mode') 
		parser.add_argument('--naive_low_resolution_test_frequency', type=int, default=1000, help='Frequency of producing test images for potential human review')
		parser.add_argument('--naive_low_resolution_save_models_frequency', type=int, default=2500, help='Frequency of saving models for reloading during later training')
		parser.add_argument('--naive_low_resolution_depth_sensor_error', type=float, default=0.33, help='End of depth loss ratio during initial naive training phase')
		parser.add_argument('--naive_low_resolution_entropy_loss_tuning_start_epoch', type=int, default=5000, help='Determine strength of entropy loss')
		parser.add_argument('--naive_low_resolution_max_entropy_weight', type=float, default=0.0001, help='Determine strength of entropy loss')
		parser.add_argument('--naive_low_resolution_pixel_samples_per_epoch', type=int, default=2048, help='The number of rows of samples to randomly collect for each image during training')
		parser.add_argument('--naive_low_resolution_number_of_test_images', type=int, default=1, help='The number of images to train on')

		# parameters for "naive" (unbiased coarse sampling) initial NeRF run at low resolution, where every pixel has a depth loss value
		parser.add_argument('--naive_high_resolution_epochs', type=int, default=1000, help='Number of epochs to run NeRF in \'naive\' coarse sampling mode at high resolution, with no constraint on ray sampling, to auto-focus on zone-of-interest voxels')
		parser.add_argument('--naive_high_resolution_number_of_images_in_training_dataset', type=int, default=512, help='Number of images to train on')
		parser.add_argument('--naive_high_resolution_image_H', type=int, default=int(480), help='Training image pixel height during \'naive\' mode') 
		parser.add_argument('--naive_high_resolution_image_W', type=int, default=int(640), help='Training image pixel width during \'naive\' mode') 
		parser.add_argument('--naive_high_resolution_test_frequency', type=int, default=1000, help='Frequency of producing test images for potential human review')
		parser.add_argument('--naive_high_resolution_save_models_frequency', type=int, default=2500, help='Frequency of saving models for reloading during later training')
		parser.add_argument('--naive_high_resolution_nerf_density_lr_end', type=float, default=0.00025, help='End of density learning rate during initial naive training phase')
		parser.add_argument('--naive_high_resolution_nerf_color_lr_end', type=float, default=0.00025, help='End of color learning rate during initial naive training phase')
		parser.add_argument('--naive_high_resolution_focal_lr_end', type=float, default=0.00001, help='End of focal learning rate during initial naive training phase')
		parser.add_argument('--naive_high_resolution_pose_lr_end', type=float, default=0.0001, help='End of pose learning rate during initial naive training phase')
		parser.add_argument('--naive_high_resolution_depth_to_rgb_loss_start', type=float, default=0.33, help='End of depth loss ratio during initial naive training phase')
		parser.add_argument('--naive_high_resolution_depth_to_rgb_loss_end', type=float, default=0.01, help='End of depth loss ratio during initial naive training phase')
		parser.add_argument('--naive_high_resolution_depth_sensor_error', type=float, default=0.33, help='End of depth loss ratio during initial naive training phase')
		parser.add_argument('--naive_high_resolution_entropy_loss_tuning_start_epoch', type=int, default=0, help='Determine strength of entropy loss')
		parser.add_argument('--naive_high_resolution_max_entropy_weight', type=float, default=0.0005, help='Determine strength of entropy loss')
		parser.add_argument('--naive_high_resolution_pixel_samples_per_epoch', type=int, default=2048, help='The number of rows of samples to randomly collect for each image during training')

		# parameters for point cloud generation for approximate mapping of object
		parser.add_argument('--point_cloud_generation_epochs', type=int, default=1, help='Number of epochs to run NeRF in \'naive\' coarse sampling mode at low resolution, with no constraint on ray sampling, to auto-focus on zone-of-interest voxels')
		parser.add_argument('--point_cloud_generation_image_H', type=int, default=int(480*1.5), help='Image height for point cloud generation (need not be too large, since goal is only to approximate voxelization)')
		parser.add_argument('--point_cloud_generation_image_W', type=int, default=int(640*1.5), help='Image width for point cloud generation (need not be too large, since goal is only to approximate voxelization)')
		parser.add_argument('--point_cloud_generation_test_frequency', type=int, default=1, help='Frequency of producing test images for potential human review')
		parser.add_argument('--point_cloud_generation_skip_training', type=bool, default=True, help='Whether to indeed skip training during testing')
		parser.add_argument('--point_cloud_generation_save_models_frequency', type=int, default=1000000, help='Frequency of saving models for reloading during later training')
		parser.add_argument('--point_cloud_generation_number_of_samples_outward_per_raycast_for_test_renders', type=int, default=500, help='Number of samples for point cloud generation (higher is better)')
		parser.add_argument('--point_cloud_generation_number_of_pixels_per_batch_in_test_renders', default=1024, type=int, help='Size in pixels of each batch input to rendering')                
		parser.add_argument('--point_cloud_generation_maximum_point_cloud_depth', type=float, default=3.0, help='Clip out all (x,y,z) points too far away from camera (increases accuracy, eliminates background)')
		parser.add_argument('--point_cloud_generation_near_maximum_depth', type=int, default=1.0, help='A percent of all raycast samples will be dedicated between the minimum depth (determined by sensor value) and this value')
		parser.add_argument('--point_cloud_generation_far_maximum_depth', type=int, default=4.0, help='The remaining percent of all raycast samples will be dedicated between the near_maximum_depth and this value')
		parser.add_argument('--point_cloud_generation_percentile_of_samples_in_near_region', type=float, default=0.90, help='This is the percent that determines the ratio between near and far sampling')

		# parameters for voxel generation for approximate mapping of object
		parser.add_argument('--point_cloud_voxelization_filename', type=str, default="all_cropped.ply", help='Name of output from merge_point_clouds()')
		parser.add_argument('--point_cloud_voxelization_prevent_visualization', type=bool, default=False, help='Whether to prevent visualization, which interrupts processing', action=argparse.BooleanOptionalAction)
		parser.add_argument('--point_cloud_voxelization_size', type=float, default=0.075, help='Size of (x,y,z) dimension for voxels')

		# parameters for "focused" voxel-based sampling of NeRF at high resolution
		parser.add_argument('--focused_high_resolution_epochs', type=int, default=35000, help='Number of epochs to run NeRF in \'focused\' coarse sampling mode at low resolution, with no constraint on ray sampling, to auto-focus on zone-of-interest voxels')
		parser.add_argument('--focused_high_resolution_number_of_images_in_training_dataset', type=int, default=512, help='Number of images to train on')
		parser.add_argument('--focused_high_resolution_image_H', type=int, default=480, help='Training image pixel height during \'focused\' mode')
		parser.add_argument('--focused_high_resolution_image_W', type=int, default=640, help='Training image pixel width during \'focused\' mode')
		parser.add_argument('--focused_high_resolution_test_frequency', type=int, default=2500, help='Frequency of producing test images for potential human review')
		parser.add_argument('--focused_high_resolution_save_models_frequency', type=int, default=5000, help='Frequency of saving models for reloading during later training')
		parser.add_argument('--focused_high_resolution_nerf_density_lr_end', type=float, default=0.0002, help='End of density learning rate during initial focused training phase')
		parser.add_argument('--focused_high_resolution_nerf_color_lr_end', type=float, default=0.0002, help='End of color learning rate during initial focused training phase')
		parser.add_argument('--focused_high_resolution_focal_lr_end', type=float, default=0.00003, help='End of focal learning rate during initial focused training phase')
		parser.add_argument('--focused_high_resolution_pose_lr_end', type=float, default=0.000100, help='End of pose learning rate during initial focused training phase')
		parser.add_argument('--focused_high_resolution_depth_to_rgb_loss_start', type=float, default=0.6, help='End of depth loss ratio during initial focused training phase')
		parser.add_argument('--focused_high_resolution_depth_to_rgb_loss_end', type=float, default=0.001, help='End of depth loss ratio during initial focused training phase')
		parser.add_argument('--focused_high_resolution_max_entropy_weight', type=float, default=0.005, help='Determine strength of entropy loss')
		parser.add_argument('--focused_high_resolution_entropy_loss_tuning_start_epoch', type=int, default=0, help='Determine strength of entropy loss')
		parser.add_argument('--focused_high_resolution_number_of_samples_outward_per_raycast', type=int, default=125, help='The number of samples per raycast to collect for each rendered pixel during training')
		parser.add_argument('--focused_high_resolution_number_of_samples_outward_per_raycast_for_test_renders', type=int, default=200, help='The number of samples per raycast to collect for each rendered pixel during testing')
		parser.add_argument('--focused_high_resolution_pixel_samples_per_epoch', type=int, default=2048, help='The number of rows of samples to randomly collect for each image during training')

		# parameters for "SDF" training
		parser.add_argument('--sdf_number_of_epochs', type=int, default=50000, help='The number of rows of samples to randomly collect for each image during training')
		parser.add_argument('--sdf_pixel_samples_per_epoch', type=int, default=int(16384*2), help='The number of rows of samples to randomly collect for each image during training')
		parser.add_argument('--sdf_number_of_pixels_per_batch_in_test_renders', default=16384*4, type=int, help='Size in pixels of each batch input to rendering')                
		parser.add_argument('--sdf_resample_pixels_frequency', default=500, type=int, help='Frequency to completely resample input training data') 

		self.args = parser.parse_args()

		if self.args.preprocess:
			self.preprocess_scan_data()

		if self.args.naive_low_resolution_training:
			# train with depth loss at low resolution
			self.neural_radiance_fields(train_mode="naive_low_resolution_training")

		if self.args.naive_high_resolution_training:
			# train with depth loss at high resolution
			self.neural_radiance_fields(train_mode="naive_high_resolution_training")

		if self.args.generate_point_clouds_from_naive_model:
			# generate point clouds used for voxel-based approximation of object location relative to pixels
			self.neural_radiance_fields(train_mode="point_cloud_generation_from_naive_model")

		if self.args.merge_naive_point_clouds:
			# combine points clouds into one
			self.merge_point_clouds(coarse_sampling_strategy="naive")

		if self.args.voxelize_point_clouds:
			# discover which voxels are most likely to include the object, to bias raycast sampling in them
			self.voxelize_object_occupancy_in_point_clouds()

		if self.args.focused_high_resolution_training:
			# train with entropy loss for M more epochs
			self.neural_radiance_fields(train_mode="focused_high_resolution_training")

		if self.args.generate_point_clouds_from_focused_model:
			self.neural_radiance_fields(train_mode="point_cloud_generation_from_focused_model")

		if self.args.merge_focused_point_clouds:
			# combine points clouds into one
			self.merge_point_clouds(coarse_sampling_strategy="focused")

		if self.args.naive_extraction_of_depth_distributions:
			self.neural_radiance_fields(train_mode="naive_extraction_of_depth_distributions")

		if self.args.sdf_training:
			self.neural_radiance_fields(train_mode="sdf")

		if self.args.export_point_clouds_of_sdf:
			self.neural_radiance_fields(train_mode="export_point_clouds_of_sdf")

		if self.args.merge_point_clouds_of_sdf:
			# combine points clouds into one
			self.merge_point_clouds(coarse_sampling_strategy="sdf", downsample_ratio=10)

		if self.args.extract_sdf_field:
			self.neural_radiance_fields(train_mode="extract_sdf")

		if self.args.export_extrinsics_intrinsics:
			self.neural_radiance_fields(train_mode="export_extrinsics_intrinsics")


	def runtime(self):
		return "{:.1f} min.".format((time.time() - self.start_time)/60)


	def preprocess_scan_data(self):		
		# infer a project name based on directory name
		self.project_name = self.args.scan_directory.split("/")[-1]

		# launch command for converting scan data into format prepared for learnign with NeRF
		print("({}) Processing raw \'{}\' 3D scan data...".format(self.runtime(), self.project_name))
		command = "python {} --dataset {} --out {}".format(self.args.scan_preprocessor, self.args.scan_directory, self.args.scan_directory)
		print("   -> {}".format(command))
		os.system(command)


	def neural_radiance_fields(self, train_mode, skip_every_n_test_images=1):
		# auto-focus is the first train mode, it features fast training at low resolution, with no assumptions on what to focus on, i.e. naive linear coarse sampling
		if train_mode == "naive_low_resolution_training":
			print("({}) Training NeRF in naive sampling mode for {} epochs at {} x {} resolution to discover zone-of-interest...".format(self.runtime(), self.args.naive_low_resolution_epochs, self.args.naive_low_resolution_image_H, self.args.naive_low_resolution_image_W))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --H_for_training {}".format(self.args.naive_low_resolution_image_H)
			command += " --W_for_training {}".format(self.args.naive_low_resolution_image_W)
			command += " --H_for_test_renders {}".format(self.args.naive_low_resolution_image_H)
			command += " --W_for_test_renders {}".format(self.args.naive_low_resolution_image_W)
			command += " --coarse_sampling_strategy naive"
			command += " --start_epoch 0"
			command += " --number_of_epochs {}".format(self.args.naive_low_resolution_epochs)
			command += " --test_frequency {}".format(self.args.naive_low_resolution_test_frequency)
			command += " --save_models_frequency {}".format(self.args.naive_low_resolution_save_models_frequency)
			command += " --entropy_loss_tuning_start_epoch {}".format(self.args.naive_low_resolution_entropy_loss_tuning_start_epoch)
			command += " --max_entropy_weight {}".format(self.args.naive_low_resolution_max_entropy_weight)
			command += " --number_of_samples_outward_per_raycast {}".format(self.args.number_of_samples_outward_per_raycast)
			command += " --number_of_samples_outward_per_raycast_for_test_renders {}".format(self.args.number_of_samples_outward_per_raycast_for_test_renders)
			command += " --pixel_samples_per_epoch {}".format(self.args.naive_low_resolution_pixel_samples_per_epoch)

			print("\n   -> {}\n".format(command))

			os.system(command)

		elif train_mode == "naive_high_resolution_training":

			print("({}) Training NeRF in naive high resolution sampling mode for {} epochs at {} x {} resolution to discover zone-of-interest...".format(self.runtime(), self.args.naive_high_resolution_epochs, self.args.naive_high_resolution_image_H, self.args.naive_high_resolution_image_W))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/naive".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --H_for_training {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_training {}".format(self.args.naive_high_resolution_image_W)
			command += " --H_for_test_renders {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_test_renders {}".format(self.args.naive_high_resolution_image_W)
			command += " --coarse_sampling_strategy naive"
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs)
			command += " --number_of_epochs {}".format(self.args.naive_high_resolution_epochs)
			command += " --test_frequency {}".format(self.args.naive_low_resolution_test_frequency)
			command += " --save_models_frequency {}".format(self.args.naive_low_resolution_save_models_frequency)
			command += " --entropy_loss_tuning_start_epoch {}".format(self.args.naive_low_resolution_entropy_loss_tuning_start_epoch)
			command += " --max_entropy_weight {}".format(self.args.naive_low_resolution_max_entropy_weight)
			command += " --number_of_samples_outward_per_raycast {}".format(self.args.number_of_samples_outward_per_raycast)
			command += " --number_of_samples_outward_per_raycast_for_test_renders {}".format(self.args.number_of_samples_outward_per_raycast_for_test_renders)
			command += " --pixel_samples_per_epoch {}".format(self.args.naive_low_resolution_pixel_samples_per_epoch)			

			print("\n   -> {}\n".format(command))

			os.system(command)


		elif train_mode == "point_cloud_generation_from_naive_model":
			print("({}) Generating point clouds from naive-trained NeRF to discover zone-of-interest...".format(self.runtime()))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --number_of_test_images {}".format(int(self.args.number_of_images_in_training_dataset))
			command += " --number_of_samples_outward_per_raycast_for_test_renders {}".format(self.args.point_cloud_generation_number_of_samples_outward_per_raycast_for_test_renders)
			command += " --number_of_pixels_per_batch_in_test_renders {}".format(self.args.point_cloud_generation_number_of_pixels_per_batch_in_test_renders)
			command += " --maximum_point_cloud_depth {}".format(self.args.point_cloud_generation_maximum_point_cloud_depth)			
			command += " --near_maximum_depth {}".format(self.args.point_cloud_generation_near_maximum_depth)
			command += " --far_maximum_depth {}".format(self.args.point_cloud_generation_far_maximum_depth)
			command += " --percentile_of_samples_in_near_region {}".format(self.args.point_cloud_generation_percentile_of_samples_in_near_region)
			command += " --skip_every_n_images_for_testing {}".format(1)
			command += " --no-train"
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/naive".format(self.args.scan_directory)
			command += " --H_for_training {}".format(self.args.point_cloud_generation_image_H)
			command += " --W_for_training {}".format(self.args.point_cloud_generation_image_W)
			command += " --H_for_test_renders {}".format(self.args.point_cloud_generation_image_H)
			command += " --W_for_test_renders {}".format(self.args.point_cloud_generation_image_W)
			command += " --coarse_sampling_strategy naive"
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs + self.args.naive_high_resolution_epochs)
			command += " --number_of_epochs {}".format(self.args.point_cloud_generation_epochs)
			command += " --test_frequency {}".format(1)
			command += " --save_models_frequency {}".format(self.args.point_cloud_generation_save_models_frequency)
			command += " --use_sparse_fine_rendering"
			command += " --save_point_clouds_during_testing"

			print("\n   -> {}\n".format(command))

			os.system(command)


		elif train_mode == "focused_high_resolution_training":
			# first derive path to voxel xyz positions that are estimated to have object inside
			path_to_voxel_xyz_file = "{}/trained_models/naive/pointclouds/{}mm_voxel_xyz_center.pt".format(self.args.scan_directory, int(self.args.point_cloud_voxelization_size * 1000))

			print("({}) Further training NeRF in focused voxel-based sampling mode for {} epochs at {} x {} resolution...".format(self.runtime(), self.args.focused_high_resolution_epochs, self.args.focused_high_resolution_image_H, self.args.focused_high_resolution_image_W))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/naive".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --H_for_training {}".format(self.args.focused_high_resolution_image_H)
			command += " --W_for_training {}".format(self.args.focused_high_resolution_image_W)
			command += " --H_for_test_renders {}".format(self.args.focused_high_resolution_image_H)
			command += " --W_for_test_renders {}".format(self.args.focused_high_resolution_image_W)
			command += " --coarse_sampling_strategy focused"
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs + self.args.naive_high_resolution_epochs)
			command += " --number_of_epochs {}".format(self.args.focused_high_resolution_epochs)
			command += " --test_frequency {}".format(self.args.focused_high_resolution_test_frequency)
			command += " --save_models_frequency {}".format(self.args.focused_high_resolution_save_models_frequency)
			command += " --nerf_density_lr_end {}".format(self.args.focused_high_resolution_nerf_density_lr_end)
			command += " --nerf_color_lr_end {}".format(self.args.focused_high_resolution_nerf_color_lr_end)
			command += " --focal_lr_end {}".format(self.args.focused_high_resolution_focal_lr_end)
			command += " --pose_lr_end {}".format(self.args.focused_high_resolution_pose_lr_end)
			command += " --depth_to_rgb_loss_start {}".format(self.args.focused_high_resolution_depth_to_rgb_loss_start)
			command += " --depth_to_rgb_loss_end {}".format(self.args.focused_high_resolution_depth_to_rgb_loss_end)
			command += " --max_entropy_weight {}".format(self.args.focused_high_resolution_max_entropy_weight)
			command += " --entropy_loss_tuning_start_epoch {}".format(self.args.focused_high_resolution_entropy_loss_tuning_start_epoch)
			command += " --object_voxel_xyz_data_file {}".format(path_to_voxel_xyz_file)
			command += " --object_voxel_size_in_meters {}".format(self.args.point_cloud_voxelization_size)
			command += " --number_of_samples_outward_per_raycast {}".format(self.args.number_of_samples_outward_per_raycast)
			command += " --number_of_samples_outward_per_raycast_for_test_renders {}".format(self.args.number_of_samples_outward_per_raycast_for_test_renders)
			command += " --pixel_samples_per_epoch {}".format(self.args.focused_high_resolution_pixel_samples_per_epoch)

			print("\n   -> {}\n".format(command))

			os.system(command)


		elif train_mode == "point_cloud_generation_from_focused_model":
			path_to_voxel_xyz_file = "{}/trained_models/naive/pointclouds/{}mm_voxel_xyz_center.pt".format(self.args.scan_directory, int(self.args.point_cloud_voxelization_size * 1000))

			print("({}) Generating point clouds from focused NeRF model for human review...".format(self.runtime()))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --number_of_test_images {}".format(int(self.args.number_of_images_in_training_dataset/4))
			command += " --number_of_samples_outward_per_raycast_for_test_renders {}".format(self.args.point_cloud_generation_number_of_samples_outward_per_raycast_for_test_renders)
			command += " --number_of_pixels_per_batch_in_test_renders {}".format(self.args.point_cloud_generation_number_of_pixels_per_batch_in_test_renders)
			command += " --maximum_point_cloud_depth {}".format(self.args.point_cloud_generation_maximum_point_cloud_depth)			
			command += " --near_maximum_depth {}".format(self.args.point_cloud_generation_near_maximum_depth)
			command += " --far_maximum_depth {}".format(self.args.point_cloud_generation_far_maximum_depth)
			command += " --percentile_of_samples_in_near_region {}".format(self.args.point_cloud_generation_percentile_of_samples_in_near_region)
			command += " --skip_every_n_images_for_testing {}".format(4)
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/focused".format(self.args.scan_directory)
			command += " --H_for_training {}".format(self.args.point_cloud_generation_image_H)
			command += " --W_for_training {}".format(self.args.point_cloud_generation_image_W)
			command += " --H_for_test_renders {}".format(self.args.point_cloud_generation_image_H)
			command += " --W_for_test_renders {}".format(self.args.point_cloud_generation_image_W)
			command += " --nerf_density_lr_end {}".format(self.args.focused_high_resolution_nerf_density_lr_end)
			command += " --nerf_color_lr_end {}".format(self.args.focused_high_resolution_nerf_color_lr_end)
			command += " --focal_lr_end {}".format(self.args.focused_high_resolution_focal_lr_end)
			command += " --pose_lr_end {}".format(self.args.focused_high_resolution_pose_lr_end)
			command += " --depth_to_rgb_loss_start {}".format(self.args.focused_high_resolution_depth_to_rgb_loss_start)
			command += " --depth_to_rgb_loss_end {}".format(self.args.focused_high_resolution_depth_to_rgb_loss_end)
			command += " --max_entropy_weight {}".format(self.args.focused_high_resolution_max_entropy_weight)
			command += " --entropy_loss_tuning_start_epoch {}".format(self.args.focused_high_resolution_entropy_loss_tuning_start_epoch)
			command += " --coarse_sampling_strategy focused"
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs + self.args.naive_high_resolution_epochs + self.args.focused_high_resolution_epochs)
			command += " --number_of_epochs {}".format(1)
			command += " --test_frequency {}".format(1)
			command += " --save_models_frequency {}".format(self.args.point_cloud_generation_save_models_frequency)
			command += " --use_sparse_fine_rendering"
			command += " --save_point_clouds_during_testing"
			command += " --object_voxel_xyz_data_file {}".format(path_to_voxel_xyz_file)
			command += " --object_voxel_size_in_meters {}".format(self.args.point_cloud_voxelization_size)
			command += " --pixel_samples_per_epoch {}".format(int(self.args.focused_high_resolution_pixel_samples_per_epoch /2))

			print("\n   -> {}\n".format(command))

			os.system(command)

		elif train_mode == "naive_extraction_of_depth_distributions":
			print("({}) Extracting from naive model depth distribution weights for subsequent SDF training...".format(self.runtime()))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --near_maximum_depth {}".format(self.args.point_cloud_generation_near_maximum_depth)
			command += " --far_maximum_depth {}".format(self.args.point_cloud_generation_far_maximum_depth)
			command += " --percentile_of_samples_in_near_region {}".format(self.args.point_cloud_generation_percentile_of_samples_in_near_region)
			command += " --skip_every_n_images_for_testing {}".format(4)
			command += " --number_of_test_images {}".format(int(self.args.number_of_images_in_training_dataset/4))
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/naive".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --H_for_training {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_training {}".format(self.args.naive_high_resolution_image_W)
			command += " --H_for_test_renders {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_test_renders {}".format(self.args.naive_high_resolution_image_W)
			command += " --coarse_sampling_strategy naive"
			command += " --extract_depth_probability_distributions"
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs + self.args.naive_high_resolution_epochs)
			command += " --number_of_epochs {}".format(1)
			command += " --test_frequency {}".format(1)	
			command += " --save_models_frequency {}".format(self.args.naive_high_resolution_save_models_frequency)
			command += " --nerf_density_lr_end {}".format(self.args.naive_high_resolution_nerf_density_lr_end)
			command += " --nerf_color_lr_end {}".format(self.args.naive_high_resolution_nerf_color_lr_end)
			command += " --focal_lr_end {}".format(self.args.naive_high_resolution_focal_lr_end)
			command += " --pose_lr_end {}".format(self.args.naive_high_resolution_pose_lr_end)
			command += " --depth_to_rgb_loss_start {}".format(self.args.naive_high_resolution_depth_to_rgb_loss_start)
			command += " --depth_to_rgb_loss_end {}".format(self.args.naive_high_resolution_depth_to_rgb_loss_end)
			command += " --max_entropy_weight {}".format(self.args.naive_high_resolution_max_entropy_weight)
			command += " --entropy_loss_tuning_start_epoch {}".format(self.args.naive_high_resolution_entropy_loss_tuning_start_epoch)
			command += " --number_of_samples_outward_per_raycast {}".format(self.args.number_of_samples_outward_per_raycast)
			command += " --number_of_samples_outward_per_raycast_for_test_renders {}".format(self.args.number_of_samples_outward_per_raycast_for_test_renders)
			command += " --pixel_samples_per_epoch {}".format(self.args.naive_high_resolution_pixel_samples_per_epoch)
			command += " --number_of_pixels_per_batch_in_test_renders {}".format(self.args.point_cloud_generation_number_of_pixels_per_batch_in_test_renders)

			print("\n   -> {}\n".format(command))

			os.system(command)

		elif train_mode == "sdf":
			print("({}) Training SDF model with NeRF depth map 16th, 50th, 84th percentiles as Bayesian priors...".format(self.runtime()))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --near_maximum_depth {}".format(self.args.point_cloud_generation_near_maximum_depth)
			command += " --far_maximum_depth {}".format(self.args.point_cloud_generation_far_maximum_depth)
			command += " --percentile_of_samples_in_near_region {}".format(self.args.point_cloud_generation_percentile_of_samples_in_near_region)
			command += " --skip_every_n_images_for_testing {}".format(4)
			command += " --number_of_test_images {}".format(int(self.args.number_of_images_in_training_dataset/4))
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/naive".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --H_for_training {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_training {}".format(self.args.naive_high_resolution_image_W)
			command += " --H_for_test_renders {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_test_renders {}".format(self.args.naive_high_resolution_image_W)
			command += " --coarse_sampling_strategy sdf"
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs + self.args.naive_high_resolution_epochs)
			command += " --number_of_epochs {}".format(self.args.sdf_number_of_epochs)
			command += " --test_frequency {}".format(1000000)	
			command += " --save_models_frequency {}".format(500)
			command += " --nerf_density_lr_end {}".format(self.args.naive_high_resolution_nerf_density_lr_end)
			command += " --nerf_color_lr_end {}".format(self.args.naive_high_resolution_nerf_color_lr_end)
			command += " --focal_lr_end {}".format(self.args.naive_high_resolution_focal_lr_end)
			command += " --pose_lr_end {}".format(self.args.naive_high_resolution_pose_lr_end)
			command += " --depth_to_rgb_loss_start {}".format(self.args.naive_high_resolution_depth_to_rgb_loss_start)
			command += " --depth_to_rgb_loss_end {}".format(self.args.naive_high_resolution_depth_to_rgb_loss_end)
			command += " --max_entropy_weight {}".format(self.args.naive_high_resolution_max_entropy_weight)
			command += " --entropy_loss_tuning_start_epoch {}".format(self.args.naive_high_resolution_entropy_loss_tuning_start_epoch)
			command += " --number_of_samples_outward_per_raycast {}".format(self.args.number_of_samples_outward_per_raycast)
			command += " --number_of_samples_outward_per_raycast_for_test_renders {}".format(self.args.number_of_samples_outward_per_raycast_for_test_renders)
			command += " --pixel_samples_per_epoch {}".format(self.args.sdf_pixel_samples_per_epoch)
			command += " --number_of_pixels_per_batch_in_test_renders {}".format(self.args.sdf_number_of_pixels_per_batch_in_test_renders)
			command += " --resample_pixels_frequency {}".format(self.args.sdf_resample_pixels_frequency)
			
			print("\n   -> {}\n".format(command))

			os.system(command)
		
		elif train_mode == "export_point_clouds_of_sdf":
			print("({}) Extracting point clouds from pre-computed SDF depth percentiles and trained colors".format(self.runtime()))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --skip_every_n_images_for_testing {}".format(4)
			command += " --number_of_test_images {}".format(int(self.args.number_of_images_in_training_dataset/4))
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/sdf".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --H_for_training {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_training {}".format(self.args.naive_high_resolution_image_W)
			command += " --H_for_test_renders {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_test_renders {}".format(self.args.naive_high_resolution_image_W)
			command += " --coarse_sampling_strategy sdf"
			command += " --focal_lr_end {}".format(self.args.naive_high_resolution_focal_lr_end)
			command += " --pose_lr_end {}".format(self.args.naive_high_resolution_pose_lr_end)
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs + self.args.naive_high_resolution_epochs + self.args.sdf_number_of_epochs)
			command += " --save_point_clouds_during_testing"
			command += " --maximum_point_cloud_depth {}".format(1.5)			
			command += " --number_of_epochs {}".format(1)
			command += " --test_frequency {}".format(1)	
			command += " --pixel_samples_per_epoch {}".format(self.args.sdf_pixel_samples_per_epoch)
			command += " --number_of_pixels_per_batch_in_test_renders {}".format(self.args.sdf_number_of_pixels_per_batch_in_test_renders)
			command += " --resample_pixels_frequency {}".format(self.args.sdf_resample_pixels_frequency)			

			print("\n   -> {}\n".format(command))

			os.system(command)

		elif train_mode == "extract_sdf":
			print("({}) Extracting SDF field...".format(self.runtime()))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --skip_every_n_images_for_testing {}".format(4)
			command += " --number_of_test_images {}".format(int(self.args.number_of_images_in_training_dataset/4))
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/sdf".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --H_for_training {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_training {}".format(self.args.naive_high_resolution_image_W)
			command += " --H_for_test_renders {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_test_renders {}".format(self.args.naive_high_resolution_image_W)
			command += " --coarse_sampling_strategy sdf"
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs + self.args.naive_high_resolution_epochs + 500) # self.args.sdf_number_of_epochs
			command += " --extract_sdf_field"
			command += " --number_of_epochs {}".format(1)
			command += " --test_frequency {}".format(1)	
			command += " --pixel_samples_per_epoch {}".format(self.args.sdf_pixel_samples_per_epoch)
			command += " --number_of_pixels_per_batch_in_test_renders {}".format(self.args.sdf_number_of_pixels_per_batch_in_test_renders)
			command += " --resample_pixels_frequency {}".format(self.args.sdf_resample_pixels_frequency)

			print("\n   -> {}\n".format(command))

			os.system(command)



		elif train_mode == "export_extrinsics_intrinsics":
			print("({}) Exporting camera extrinsics and intrinsics...".format(self.runtime()))
			command = "python nerf.py"
			command += " --base_directory {}".format(self.args.scan_directory)
			command += " --export_extrinsics_intrinsics"
			command += " --near_maximum_depth {}".format(self.args.point_cloud_generation_near_maximum_depth)
			command += " --far_maximum_depth {}".format(self.args.point_cloud_generation_far_maximum_depth)
			command += " --percentile_of_samples_in_near_region {}".format(self.args.point_cloud_generation_percentile_of_samples_in_near_region)
			command += " --skip_every_n_images_for_testing {}".format(4)
			command += " --number_of_test_images {}".format(int(self.args.number_of_images_in_training_dataset/4))
			command += " --load_pretrained_models"
			command += " --pretrained_models_directory {}/trained_models/naive".format(self.args.scan_directory)
			command += " --number_of_images_in_training_dataset {}".format(self.args.number_of_images_in_training_dataset)
			command += " --H_for_training {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_training {}".format(self.args.naive_high_resolution_image_W)
			command += " --H_for_test_renders {}".format(self.args.naive_high_resolution_image_H)
			command += " --W_for_test_renders {}".format(self.args.naive_high_resolution_image_W)
			command += " --coarse_sampling_strategy naive"
			command += " --start_epoch {}".format(self.args.naive_low_resolution_epochs + self.args.naive_high_resolution_epochs)
			command += " --number_of_epochs {}".format(1)
			command += " --test_frequency {}".format(1)	
			command += " --save_models_frequency {}".format(self.args.naive_high_resolution_save_models_frequency)
			command += " --nerf_density_lr_end {}".format(self.args.naive_high_resolution_nerf_density_lr_end)
			command += " --nerf_color_lr_end {}".format(self.args.naive_high_resolution_nerf_color_lr_end)
			command += " --focal_lr_end {}".format(self.args.naive_high_resolution_focal_lr_end)
			command += " --pose_lr_end {}".format(self.args.naive_high_resolution_pose_lr_end)
			command += " --depth_to_rgb_loss_start {}".format(self.args.naive_high_resolution_depth_to_rgb_loss_start)
			command += " --depth_to_rgb_loss_end {}".format(self.args.naive_high_resolution_depth_to_rgb_loss_end)
			command += " --max_entropy_weight {}".format(self.args.naive_high_resolution_max_entropy_weight)
			command += " --entropy_loss_tuning_start_epoch {}".format(self.args.naive_high_resolution_entropy_loss_tuning_start_epoch)
			command += " --number_of_samples_outward_per_raycast {}".format(self.args.number_of_samples_outward_per_raycast)
			command += " --number_of_samples_outward_per_raycast_for_test_renders {}".format(self.args.number_of_samples_outward_per_raycast_for_test_renders)
			command += " --pixel_samples_per_epoch {}".format(self.args.naive_high_resolution_pixel_samples_per_epoch)
			command += " --number_of_pixels_per_batch_in_test_renders {}".format(self.args.point_cloud_generation_number_of_pixels_per_batch_in_test_renders)

			print("\n   -> {}\n".format(command))

			os.system(command)


			
	def merge_point_clouds(self, coarse_sampling_strategy="naive", downsample_ratio=1):		
		point_cloud_directory = "{}/trained_models/{}/pointclouds".format(self.args.scan_directory, coarse_sampling_strategy)

		# launch command for converting scan data into format prepared for learning with NeRF
		print("({}) Merging point clouds in {}...".format(self.runtime(), point_cloud_directory))

		command = "python merge_point_clouds.py --point_clouds_directory {} --downsample_ratio {}".format(point_cloud_directory, downsample_ratio)
		print("   -> {}".format(command))
		
		os.system(command)


	def voxelize_object_occupancy_in_point_clouds(self):
		merged_point_cloud_path = "{}/trained_models/naive/pointclouds/{}".format(self.args.scan_directory, self.args.point_cloud_voxelization_filename)

		# launch command for voxelizing point clouds in areas of estimated object occupancy for ~10x faster higher resolution training
		print("({}) Voxelizing point cloud {} to extract estimated object occupancy for ~10x faster higher resolution training...".format(self.runtime(), merged_point_cloud_path))

		command =  "python voxelize_point_clouds.py"
		command += " --path_to_cropped_merged_point_cloud {}".format(merged_point_cloud_path)

		if self.args.point_cloud_voxelization_prevent_visualization:
			command += " --prevent_visualization_of_extracted_voxels"
		else:
			command += " --no-prevent_visualization_of_extracted_voxels"

		command += " --xyz_voxel_size {}".format(self.args.point_cloud_voxelization_size)
		
		print("   -> {}".format(command))
		
		os.system(command)


if __name__ == "__main__":
	Replicator()