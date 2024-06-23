import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
import math
from scanning import *
import os

big_board = True
if big_board:
    aruco_dict_selection = aruco.DICT_5X5_1000
    square_length = 40
    marker_length = 33
    board_squares_x = 27
    board_squares_y = 35
    size_of_marker = marker_length

else:
    aruco_dict_selection = aruco.DICT_5X5_1000
    square_length = 65
    marker_length = 50
    size_of_marker = 50
    board_squares_x = 126
    board_squares_y = 7
    size_of_marker = marker_length

camera_distortion_coefficients = np.asarray([-2.0596606193800684e-01, 1.8156023227741025e-01, 7.1149635146760312e-04, 7.4055906083523345e-04, -5.7471740977204609e-02])

camera_intrinsics_matrix = np.asarray(  [[2.3396800035350443e+03,    0.0,                    1.0880643451058950e+03  ],
                                        [0.0,                       2.3331185597670483e+03, 1.0568206694191294e+03  ],
                                        [0.0,                       0.0,                    1.0                     ]])






aruco_dict = aruco.Dictionary_get(aruco_dict_selection)
board = cv2.aruco.CharucoBoard_create(board_squares_x, board_squares_y, square_length/1000, marker_length/1000, aruco_dict)

def export_reindexed_detections_per_frame(detection_dictionary):
    pass

def calibrate_camera(all_coordinates, all_ids, image_size):
    #+ cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_ASPECT_RATIO
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS ) #+ cv2.CALIB_RATIONAL_MODEL ) # + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_P1 + cv2.CALIB_FIX_P2 + cv2.CALIB_FIX_ASPECT_RATIO
    print("Optimizing instrinsic + extrinsic camera calibration...")
    ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=all_coordinates,
                                                                                                                                                                                                    charucoIds=all_ids,
                                                                                                                                                                                                    board=board,
                                                                                                                                                                                                    imageSize=(image_size[0], image_size[1]),
                                                                                                                                                                                                    cameraMatrix=camera_intrinsics_matrix,
                                                                                                                                                                                                    distCoeffs=camera_distortion_coefficients,
                                                                                                                                                                                                    flags=flags,
                                                                                                                                                                                                    criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print("{} average pixel error".format(ret))
    print("CAM MATRIX: {}".format(camera_matrix))
    print("DISTORTION COEF: {}".format(distortion_coefficients0))
    print("ROTATION VECS: {}".format(rotation_vectors))
    print("TRANSLATION VECS: {}".format(translation_vectors))
    print("STDEV INTRINSICS: {}".format(stdDeviationsIntrinsics))
    print("STDEV EXTRINSiCS: {}".format(stdDeviationsExtrinsics))
    print("PER VIEW ERRORS: {}".format(perViewErrors))

    return rotation_vectors, translation_vectors


def find_charuco_poses(images):
    all_corner_coordinates = []
    all_corner_ids = []
    image_size = None
    for image in images:
        corner_coordinates, corner_ids, image_size, frame = detect_saddle_points(image=image)
        all_corner_coordinates.append(corner_coordinates)
        all_corner_ids.append(corner_ids)
        image_size = image_size

    return calibrate_camera(all_corner_coordinates, all_corner_ids, image_size)

def find_charuco_pose(image, plot_axes=False):
    tag_name = image.rstrip(".png")  
    corner_coordinates, corner_ids, image_size, frame = detect_saddle_points(image=image)
    if type(corner_coordinates) != type(None) and len(corner_coordinates) >= 4:
        rotation_vector = np.asarray([[0.0], [0.0], [0.0]])
        translation_vector = np.asarray([[0.0], [0.0], [0.0]])
        fit_value, rotation_vector, translation_vector = aruco.estimatePoseCharucoBoard(corner_coordinates, corner_ids, board, camera_intrinsics_matrix, camera_distortion_coefficients, rotation_vector, translation_vector, useExtrinsicGuess=False)
        if plot_axes:
            aruco.drawAxis(frame, camera_intrinsics_matrix, camera_distortion_coefficients, rotation_vector, translation_vector, 0.3)
            plt.imshow(frame, interpolation = "nearest")
            plt.tight_layout()
            plt.savefig("{}_with_detections_and_pose.png".format(tag_name))

            plt.show()
        translation_vector = np.asarray(translation_vector).flatten().tolist()
        return fit_value, rotation_vector, translation_vector
    else:
        return False, None, None


def detect_saddle_points(image, show_possible_detections=False, show_aruco_detections=True, show_final_detections=True):
    print(image)
    frame = cv2.imread(image) 
    #frame = cv2.flip(frame, 1) # horizontal flip
    plt.figure(figsize=(20, 20))

    undistort = True
    if undistort:
        print("Undistorting image...")
        frame = cv2.undistort(src = frame, cameraMatrix = camera_intrinsics_matrix, distCoeffs = camera_distortion_coefficients)
        plt.imshow(frame, interpolation = "nearest")
        plt.show()
        new_filename = image.replace(".png","_undistorted.png")
        cv2.imwrite(new_filename, frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000) #aruco.DICT_ARUCO_ORIGINAL
    parameters =  aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 150
    parameters.adaptiveThreshWinSizeStep = 3
    parameters.minCornerDistanceRate = 0.01
    parameters.minMarkerDistanceRate = 0.01
    # parameters.adaptiveThreshConstant = 10
    # parameters.minMarkerPerimeterRate = 0.1
    # parameters.maxMarkerPerimeterRate = 10.0
    # parameters.polygonalApproxAccuracyRate = 0.1
    # parameters.minDistanceToBorder = 3
    # parameters.minOtsuStdDev = 100.0
    # parameters.perspectiveRemovePixelPerCell = 1
    # parameters.perspectiveRemoveIgnoredMarginPerCell = 0.4
    # parameters.maxErroneousBitsInBorderRate = 0.5
    # parameters.errorCorrectionRate = 0.5
    # parameters.cornerRefinementMethod = 0
    # parameters.cornerRefinementMaxIterations = 1000
    # parameters.cornerRefinementWinSize = 5

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    #if show_possible_detections or show_aruco_detections or show_final_detections:


    if show_possible_detections:
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), rejectedImgPoints, np.asarray([np.float32(i) for i, thing in enumerate(rejectedImgPoints)]))
        plt.imshow(frame_markers, interpolation = "nearest")
        plt.show()

    # SUB PIXEL DETECTION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000000, 0.001)
    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize = (5,5), zeroZone = (3,3), criteria = criteria)

    if len(corners) == 0:
        print("Failure to find ARUCO markers")
        return None, None, None, None

    if show_aruco_detections:
        rotation_vectors, translation_vectors, board_relative_position_vectors = aruco.estimatePoseSingleMarkers(corners, size_of_marker, camera_intrinsics_matrix, camera_distortion_coefficients)
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        length_of_axis = 0.1
        for i in range(len(translation_vectors)):
            imaxis = aruco.drawAxis(frame_markers, camera_intrinsics_matrix, camera_distortion_coefficients, rotation_vectors[i], translation_vectors[i], length_of_axis)
        plt.imshow(frame_markers, interpolation = "nearest")
        plt.show()

    imboard = board.draw((2000, 2000))

    charuco_corners = np.array([])
    charuco_ids = np.array([])

    if (ids is not None):
        corner_count, corner_coordinates, corner_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
        print("Found {} corners".format(corner_count)) 

        if corner_ids is not None:
            charuco_corners = corner_coordinates[:, 0]
            charuco_ids = corner_ids[:, 0]

        if charuco_ids.size > 0:
            rect_size = 5
            id_font = cv2.FONT_HERSHEY_DUPLEX
            id_scale = 0.2
            id_color = (0, 255, 0)
            rect_thickness = 1

            for (corner, id) in zip(charuco_corners, charuco_ids):
                corner_x = int(corner[0])
                corner_y = int(corner[1])
                id_text = "{}".format(str(id))
                id_coord = (corner_x + 2*rect_size, corner_y + 2*rect_size)
                cv2.rectangle(frame, (corner_x - rect_size, corner_y - rect_size),
                            (corner_x + rect_size, corner_y + rect_size),
                            id_color, thickness=rect_thickness)
                cv2.putText(frame, id_text, id_coord, id_font, id_scale, id_color)
                #aruco.drawDetectedCornersCharuco(frame.copy(), corners, ids)

        if show_final_detections:
            plt.imshow(frame, interpolation = "nearest")
            plt.show()

        image_size = frame.shape

        # scaling an image
        # scale_percent = 500 # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        return corner_coordinates, corner_ids, image_size, frame
    else:
        return None, None, None, None


# def find_all_saddle_points_for_directory(project_directory):
#     # all_files_in_directory = os.listdir(project_directory)
#     # images = [f for f in all_files_in_directory if ".png" in f]
#     # images.sort()
#     # parses = [f.split(tag)[1].split("_")[1].split("view")[1] for f in images]
#     # view_indices = np.asarray([int(parse) for parse in parses])
#     # sorted_indices = np.argsort(view_indices)

#     # for view_index in range(1,total_views+1):
#     #     next_view = np.where(view_indices == view_index)
#     #     next_image = images[int(next_view[0])]
#     #     all_images.append(next_image)
#     #     full_path = "{}/{}".format(project_directory, next_image)

if __name__ == "__main__":
    structured_scan_processing = False
    if structured_scan_processing:
        working_directory = "/home/sense/3cobot"
        project_name = "sfm_v2_"
        scan_data = {}

        min_scan_index = 1
        max_image_index = 0
        max_cloud_index = 0

        png_images = [f for f in os.listdir(working_directory) if project_name in f and ".png" in f]
        for image in png_images:
            i = int(image.split(project_name)[1].split("view")[1].split("_")[0])
            scan_data["{}_image".format(i)] = image
            #print("({}) {}".format(i, image))
            x = float(image.split("_x")[1].split("_y")[0])
            y = float(image.split("_y")[1].split("_z")[0])
            z = float(image.split("_z")[1].split("_yaw")[0])
            yaw = float(image.split("_yaw")[1].split("_pitch")[0])
            pitch = float(image.split("_pitch")[1].split("_rgb")[0])
            #print("({}) (x,y,z,pitch,yaw)=({:.3f},{:.3f},{:.3f},{:.3f},{:.3f})".format(i,x,y,z,pitch,yaw))
            scan_data["{}_x".format(i)] = x
            scan_data["{}_y".format(i)] = y
            scan_data["{}_z".format(i)] = z
            scan_data["{}_pitch".format(i)] = pitch
            scan_data["{}_yaw".format(i)] = yaw
            if i > max_image_index:
                max_image_index = i

        ply_clouds = [f for f in os.listdir(working_directory) if project_name in f and ".ply" in f and "union" not in f]
        for cloud in ply_clouds:
            i = int(cloud.split(project_name)[1].split("view")[1].split("_")[0])
            scan_data["{}_cloud".format(i)] = cloud
            #print("({}) {}".format(i, cloud))
            if i > max_cloud_index:
                max_cloud_index = i

        if max_image_index <= max_cloud_index:
            max_index = max_image_index
        else:
            max_index = max_cloud_index

        print()
        print()

        all_images = []
        all_clouds = []

        for i in range(min_scan_index, max_index+1):
            all_images.append(scan_data["{}_image".format(i)])
            all_clouds.append(scan_data["{}_cloud".format(i)])

            print(scan_data["{}_image".format(i)])
            #print(scan_data["{}_cloud".format(i)])


        sample_size = 100
        images = all_images[:sample_size]
        cloud_names = all_clouds[:sample_size]

        use_precomputed_translations_rotations = False
        camera_calibrate_charuco = False
        calibrate_stereo = False

        # 1 then 2
        detect_charuco_only = True
        parse_sfm_localizations = False

        if detect_charuco_only:
             # Key data structures to enable
             # point_1 : x1 y1 x2 y2 ... xTOTAL_SEEN_IMAGES yTOTAL_SEEN_IMAGES\n
             # point_2 : x1 y1 x2 y2 ... xTOTAL_SEEN_IMAGES yTOTAL_SEEN_IMAGES\n
             # ...
             # point_N : x1 y1 x2 y2 ... xTOTAL_SEEN_IMAGES yTOTAL_SEEN_IMAGES\n

            max_point_index = 0
            point_index_map = {} # from raw board-based identifier to our own global identifier
            points_in_frames = [[] for image in images] # list of lists, where each inner list is a normalized point index
            normalized_point_to_x_per_image = [{} for image in images]
            normalized_point_to_y_per_image = [{} for image in images]

            print("Finding ChAruCo poses for images")
            for image_number, image in enumerate(images):
                print("Image {}".format(image_number))
                corner_coordinates, corner_ids, _, _ = detect_saddle_points(image)
                print(corner_coordinates)
                print()
                print(corner_ids)
                for corner_id, corner_coordinate in zip(corner_ids, corner_coordinates):
                    corner_id = int(corner_id[0])
                    x = float(corner_coordinate[0][0])
                    y = float(corner_coordinate[0][1])
                    normalized_id = point_index_map.get(corner_id, max_point_index)
                    if normalized_id == max_point_index:
                        point_index_map[corner_id] = max_point_index
                        max_point_index += 1
                        # update points_in_frames for previous images to -1,-1
                    print("(NORM ID={} ({})) := (x,y)=({},{})".format(normalized_id, corner_id, x, y))
                    normalized_point_to_x_per_image[image_number][normalized_id] = x
                    normalized_point_to_y_per_image[image_number][normalized_id] = y
                    points_in_frames[image_number].append(normalized_id)

            with open("{}/{}localizations.txt".format(working_directory,project_name), "w") as output_file:
                for point_index in range(max_point_index):
                    output_line = ""
                    for image_number, image in enumerate(images):
                        if point_index in points_in_frames[image_number]:
                            x = normalized_point_to_x_per_image[image_number][point_index]
                            y = normalized_point_to_y_per_image[image_number][point_index]
                        else:
                            x = "-1.00"
                            y = "-1.00"
                        output_line = ("{} {} {}").format(output_line, x, y)
                    output_line.lstrip()
                    if point_index == max_point_index - 1:
                        output_line = ("{}").format(output_line)
                    else:
                        output_line = ("{}\n").format(output_line)
                    output_file.write(output_line)

            print(normalized_point_to_x_per_image)
            print(normalized_point_to_x_per_image)
            print(points_in_frames)
            print(max_point_index)
            # Currently you are trying to save a global frame mapping where -1 -1 shows up for normalized values that do not appear in this image


            # translations = []
            #     for raw_translation in raw_translations:
            #         listed_translation = np.asarray(raw_translation).tolist()
            #         listed_translation = np.asarray([[translation[0] * 1000] for translation in listed_translation])
            #         translations.append(listed_translation)

            #     with open('{}rotations.npy'.format(project_name), 'wb') as output_file:
            #         np.save(output_file, rotations)
            #     with open('{}translations.npy'.format(project_name), 'wb') as output_file:
            #         np.save(output_file, translations)


        elif camera_calibrate_charuco:
            if use_precomputed_translations_rotations:
                with open('{}rotations.npy'.format(project_name), 'rb') as input_file:
                    rotations = np.load(input_file)
                with open('{}translations.npy'.format(project_name), 'rb') as input_file:
                    translations = np.load(input_file)
            else:
                print("Finding ChAruCo poses for images")
                rotations, raw_translations = find_charuco_poses(images)
                translations = []
                for raw_translation in raw_translations:
                    listed_translation = np.asarray(raw_translation).tolist()
                    listed_translation = np.asarray([[translation[0] * 1000] for translation in listed_translation])
                    translations.append(listed_translation)

                with open('{}rotations.npy'.format(project_name), 'wb') as output_file:
                    np.save(output_file, rotations)
                with open('{}translations.npy'.format(project_name), 'wb') as output_file:
                    np.save(output_file, translations)
        elif calibrate_stereo:
            if use_precomputed_translations_rotations:
                pass
                # with open('{}rotations.npy'.format(project_name), 'rb') as input_file:
                #     rotations = np.load(input_file)
                # with open('{}translations.npy'.format(project_name), 'rb') as input_file:
                #     translations = np.load(input_file)
            else:
                print("Finding rotation and transformation between two camera poses")
                pass
                # rotations, raw_translations = find_charuco_poses(images)
                # translations = []
                # for raw_translation in raw_translations:
                #     listed_translation = np.asarray(raw_translation).tolist()
                #     listed_translation = [[translation[0] * 1000] for translation in listed_translation]
                #     translations.append(listed_translation)

                # with open('{}rotations.npy'.format(project_name), 'wb') as output_file:
                #     np.save(output_file, rotations)
                # with open('{}translations.npy'.format(project_name), 'wb') as output_file:
                #     np.save(output_file, translations)       
        elif parse_sfm_localizations:
            processed_clouds = []
            for camera_number in range(sample_size):
                camera_rotations = cv2.FileStorage("/home/sense/camera_rotations.xml", cv2.FILE_STORAGE_READ)
                rotations = camera_rotations.getNode("camera_rotations").at(camera_number).mat()

                camera_translations = cv2.FileStorage("/home/sense/camera_translations.xml", cv2.FILE_STORAGE_READ)
                translations = camera_translations.getNode("camera_translations").at(camera_number).mat()

                print("Rotations:")
                print(rotations)
                print()
                print("Translations")
                print(translations)

                print("Loading point cloud...")
                cloud_name = cloud_names[camera_number]
                cloud = PointCloud(filename="/home/sense/3cobot/{}".format(cloud_name))
                cloud.downsample(reduce_by_1_divided_by_n=50)
                print("Transforming points")
                cloud.homogeneous_transform(translation_matrix=translations, rotation_matrix=rotations)
                #cloud.homogeneous_transform_inverted(translation_vector=translations, rotation_matrix=rotations)
                processed_clouds.append(cloud)

            print("Combining point clouds into single one")
            master_cloud = processed_clouds[0]
            for cloud in processed_clouds[1:]:
                master_cloud.add_point_cloud(cloud)




            # So, at this point all of the key infrastructure is in place, including a recent 100x scan of "feature points" recognized across 2D images...
            # The OpenCV SFM module is working, but whether it is performing correctly, and whether the units for rotations and translations are properly handled is unknown
            # Multiplications of these outputs are proceeding in homogeneous coordinate systems, but alternatively, it may be necessary to trace through some math...
            # As well, is OpenCV SFM really the best module to do the job?

            print("Exporting to CSV")
            master_cloud.export_to_csv("{}union".format(project_name))
        else:
            print("No calibration technique specified")
            sys.exit(0)

    else:
        images = ["/home/sense/3cobot/analysis/3d_hdr_v2.png", "/home/sense/3cobot/analysis/3d_hdr_v2_rgb.png"]
        for image in images:
            print(image)
            corner_coordinates, corner_ids, _, _ = detect_saddle_points(image)
            print(corner_ids)
            print(corner_coordinates)


  