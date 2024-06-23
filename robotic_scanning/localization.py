import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import sys
import math
import os

camera_width = 2048
camera_height = 2048

big_board = False
curved_board = True
if big_board:
    aruco_dict_selection = aruco.DICT_5X5_1000
    square_length = 40
    marker_length = 33
    board_squares_x = 27
    board_squares_y = 35
    size_of_marker = marker_length

elif curved_board:
    aruco_dict_selection = aruco.DICT_APRILTAG_36h10
    square_length = 32
    marker_length = 21
    board_squares_x = 15
    board_squares_y = 48
    size_of_marker = marker_length

    number_of_planes = 12
    number_of_points_per_plane = 42
    number_of_columns_per_row = 14
    number_of_rows_per_plane = 3

    curved_board_planes = {}
    curved_board_planes[0] = [i for i in range(0,42)]
    curved_board_planes[1] = [i for i in range(56,98)]
    curved_board_planes[2] = [i for i in range(112,154)]
    curved_board_planes[3] = [i for i in range(168,210)]
    curved_board_planes[4] = [i for i in range(224,266)]
    curved_board_planes[5] = [i for i in range(280,322)]
    curved_board_planes[6] = [i for i in range(336,378)]
    curved_board_planes[7] = [i for i in range(392,434)]
    curved_board_planes[8] = [i for i in range(448,490)]
    curved_board_planes[9] = [i for i in range(504,546)]
    curved_board_planes[10] = [i for i in range(560,602)]
    curved_board_planes[11] = [i for i in range(616,658)]

    assert(len(curved_board_planes[0]) == len(curved_board_planes[1]))
    assert(len(curved_board_planes[1]) == len(curved_board_planes[2]))
    assert(len(curved_board_planes[2]) == len(curved_board_planes[3]))
    assert(len(curved_board_planes[3]) == len(curved_board_planes[4]))
    assert(len(curved_board_planes[4]) == len(curved_board_planes[5]))
    assert(len(curved_board_planes[5]) == len(curved_board_planes[6]))
    assert(len(curved_board_planes[6]) == len(curved_board_planes[7]))
    assert(len(curved_board_planes[7]) == len(curved_board_planes[8]))
    assert(len(curved_board_planes[8]) == len(curved_board_planes[9]))
    assert(len(curved_board_planes[9]) == len(curved_board_planes[10]))
    assert(len(curved_board_planes[10]) == len(curved_board_planes[11]))

    charuco_id_to_planes = {}
    for plane_number, charuco_ids in curved_board_planes.items():
        for charuco_id in charuco_ids:
            charuco_id_to_planes[charuco_id] = plane_number

    charuco_id_to_plane_point_index = {}
    for plane_number, charuco_ids in curved_board_planes.items():
        for plane_point_index, charuco_id in enumerate(charuco_ids):
            charuco_id_to_plane_point_index[charuco_id] = plane_point_index

    distance_matrix = np.zeros((number_of_points_per_plane, number_of_points_per_plane))

    column_positions = np.linspace(0, square_length*(number_of_columns_per_row-1), number_of_columns_per_row)
    row_positions = np.linspace(0, square_length*(number_of_rows_per_plane-1), number_of_rows_per_plane)

    for row_1 in range(number_of_rows_per_plane):
        row_1_position = row_positions[row_1]
        for column_1 in range(number_of_columns_per_row):
            column_1_position = column_positions[column_1]
            point_1_index = int(row_1*number_of_columns_per_row + column_1)
            for row_2 in range(number_of_rows_per_plane):
                row_2_position = row_positions[row_2]
                for column_2 in range(number_of_columns_per_row):
                    column_2_position = column_positions[column_2]
                    point_2_index = int(row_2*number_of_columns_per_row + column_2)
                    distance = math.sqrt((row_2_position - row_1_position)**2 + (column_2_position-column_1_position)**2)
                    distance_matrix[point_1_index, point_2_index] = round(distance,4)
    plane_distance_matrix = distance_matrix
    # print("Distance matrix:")
    # print(distance_matrix)

    #for plane_number in range(number_of_planes):
    #    charuco_ground_truth_distance_per_plane[plane_number,:,:] = distance_matrix

    dead_row_1 = [602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615]
    dead_row_2 = [546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559]
    dead_row_3 = [490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503]
    dead_row_4 = [434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447]
    dead_row_5 = [378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391]
    dead_row_6 = [322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335]
    dead_row_7 = [266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279]
    dead_row_8 = [210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223]
    dead_row_9 = [154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167]
    dead_row_10 = [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    dead_row_11 = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    dead_corners = dead_row_11 + dead_row_10 + dead_row_9 + dead_row_8 + dead_row_7 + dead_row_6 + dead_row_5 + dead_row_4 +  dead_row_3 +  dead_row_2 +  dead_row_1

else:
    aruco_dict_selection = aruco.DICT_5X5_1000
    square_length = 65
    marker_length = 50
    size_of_marker = 50
    board_squares_x = 126
    board_squares_y = 7
    size_of_marker = marker_length

aruco_dict = aruco.Dictionary_get(aruco_dict_selection)
board = cv2.aruco.CharucoBoard_create(board_squares_x, board_squares_y, square_length/1000, marker_length/1000, aruco_dict)


camera_distortion_coefficients = np.asarray([-2.0596606193800684e-01, 1.8156023227741025e-01, 7.1149635146760312e-04, 7.4055906083523345e-04, -5.7471740977204609e-02])

camera_intrinsics_matrix = np.asarray(  [[2.3396800035350443e+03,    0.0,                    1.0880643451058950e+03  ],
                                        [0.0,                       2.3331185597670483e+03, 1.0568206694191294e+03  ],
                                        [0.0,                       0.0,                    1.0                     ]])


def get_charuco_id_to_plane_data():
    return charuco_id_to_planes, charuco_id_to_plane_point_index, plane_distance_matrix, curved_board_planes

def undistortion_with_intrinsics(image, short_way= True, cropping=False, show_plot=False, new_camera_intrinsics=True):
    if new_camera_intrinsics:
        new_camera_intrinsics_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_intrinsics_matrix, camera_distortion_coefficients, (camera_width,camera_height), 1, (camera_width,camera_height))
    else:
        new_camera_intrinsics_matrix = camera_intrinsics_matrix

    original_frame = cv2.imread(image) 

    if short_way:
        remapped_image = cv2.undistort(src = original_frame, cameraMatrix = camera_intrinsics_matrix, distCoeffs = camera_distortion_coefficients, newCameraMatrix=new_camera_intrinsics_matrix)
    else:
        mapx,mapy = cv2.initUndistortRectifyMap(camera_intrinsics_matrix, camera_distortion_coefficients, None, new_camera_intrinsics_matrix, (camera_width,camera_height), 5)
        remapped_image = cv2.remap(original_frame, mapx, mapy, cv2.INTER_LINEAR)
    
    if cropping:
        x,y,w,h = roi
        remapped_image = remapped_image[y:y+h, x:x+w]

    if show_plot:
        plt.figure(figsize=(20, 20))
        plt.imshow(original_frame, interpolation = "nearest")
        plt.show()
        plt.imshow(remapped_image, interpolation = "nearest")
        plt.show()

    f_x = new_camera_intrinsics_matrix[0][0]
    f_y = new_camera_intrinsics_matrix[1][1]
    c_x = new_camera_intrinsics_matrix[0][2]
    c_y = new_camera_intrinsics_matrix[1][2]

    return f_x, f_y, c_x, c_y, remapped_image


def xyz_pixel_forward_projection_from_focal_point(u, v, f_x, f_y, c_x, c_y, z=1):
    # solve for (x,y,z) in (e.g.) normalized camera coordinates
    x =      (v - c_x) * z / f_x 
    y = -1 * (u - c_y) * z / f_y   #f_y
    z = z
    #z = z

    # print("From (u,v)=({:.3f},{:3f}) to (x,y,z)=({:.3f},{:.3f},{:.3f})".format(u,v,x,y,z))


    # #translation_matrix = np.asarray([-1.35974633e-02,    -4.05324213e-02,       1.99965611e-02  ])
    # camera_extrinsic_rotation_vector = np.asarray([-2.51618317e+02,    -2.65178162e+02,       5.58995972e+02  ])

    # rotation_matrix = np.zeros(shape=(3,3))
    # cv2.Rodrigues(camera_extrinsic_rotation_vector, rotation_matrix)

    # all_points = np.matrix([[x],[y],[z]])
    # all_points_rotated = rotation_matrix * all_points

    # print(all_points_rotated)

    # x = np.asarray(all_points_rotated[0])[0][0]
    # y = np.asarray(all_points_rotated[1])[0][0]
    # z = np.asarray(all_points_rotated[2])[0][0]

    # print(x)

    # print("Then after rotation... to (x,y,z)=({:.3f},{:.3f},{:.3f})".format(x,y,z))

    return x,y,z 






        # camera_intrinsics_matrix = np.asarray(    [[2.3396800035350443e+03,    0.0,                    1.0880643451058950e+03  ],
        #                                       [0.0,                       2.3331185597670483e+03, 1.0568206694191294e+03  ],
        #                                       [0.0,                       0.0,                    1.0                     ]])

    # f_x = camera_intrinsics_matrix[0][0]
    # f_y = camera_intrinsics_matrix[1][1]
    # c_x = camera_intrinsics_matrix[0][2]
    # c_y = camera_intrinsics_matrix[1][2]

    # # print("f_x = {}, f_y = {}, c_x = {}, c_y = {}".format(f_x, f_y, c_x, c_y))

    # #f_x, f_y, c_x, c_y = undistortion_with_intrinsics(image_name)
    # #undistorted_image_name = image_name.replace(".png","_undistorted_with_new_intrinsics.png")

    # # Without recropping, for # 378
    # # From (u,v)=(638.844,411.929321) to (x,y,z)=(-0.310,0.193,1.000)
    # # Normalized coordinate projection (x,y,z) = (-137.65676708647533,85.48782484742189,444.0179039715645)

#cv2.Rodrigues(camera_extrinsic_rotation_vector, rotation_matrix)

#camera_extrinsic_rotation_vector = np.asarray([-2.51618317e+02,    -2.65178162e+02,       5.58995972e+02  ])
#rotation_matrix = np.zeros(shape=(3,3))


    # if extrinsic_transform:
    #     points.transform(rotation_matrix=rotation_matrix, translation_matrix=translation_matrix)


# add_point_cloud = False

    # if add_point_cloud:
    #     other_points_to_add = PointCloud(filename="{}-0-0.ply".format(point_cloud_to_add_project_name), project_name=point_cloud_to_add_project_name)
    #     points.add_point_cloud(other_points_to_add)
    #     visualization = "{}_combined_with_{}.ply".format(project_name, point_cloud_to_add_project_name)
    #     points.save_as_ply(visualization)
    #     sys.exit(0)


# camera_extrinsics_matrix = np.asarray(  [[-1.35974633e-02,    -4.05324213e-02,       1.99965611e-02  ],
#                                          [-2.51618317e+02,    -2.65178162e+02,       5.58995972e+02  ]])

# translation_matrix = np.asarray([1.94323429, -0.09890449, -0.18244184])

# rotation_matrix = np.asarray(  [[ 9.97950716e-01, 2.98418020e-04, -6.39865571e-02],
#                                 [-3.23156443e-04, 9.99999877e-01, -3.76269975e-04],
#                                 [ 6.39864370e-02, 3.96176559e-04, 9.97950690e-01]])

        # with open('{}_camera_intrinsics.npy'.format(project_name), 'wb') as output_file:
        #   np.save(output_file, camera_intrinsics_matrix)

            #     with open('{}_camera_intrinsics.npy'.format(project_name), 'rb') as input_file:
            # camera_intrinsics_matrix = np.load(input_file)


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

    # z = 1
    # x = (u-cx)*z/fx
    # y = (v-cy)*z/fy

    # # if image at 90 degrees
    # if type(camera_extrinsics_matrix) != type(None):
    #     r_x = camera_extrinsics_matrix[0][0]
    #     r_y = camera_extrinsics_matrix[0][1]
    #     r_z = camera_extrinsics_matrix[0][2]
    #     t_x = camera_extrinsics_matrix[1][0]
    #     t_y = camera_extrinsics_matrix[1][1]
    #     t_z = camera_extrinsics_matrix[1][2]


    # camera_extrinsics_matrix = np.asarray(  [[-1.35974633e-02,    -4.05324213e-02,       1.99965611e-02  ],
    #                                          [-2.51618317e+02,    -2.65178162e+02,       5.58995972e+02  ]])

    # https://answers.opencv.org/question/4862/how-can-i-do-back-projection/
    # https://answers.opencv.org/question/117354/back-projecting-a-2d-point-to-a-ray/
    # https://en.wikipedia.org/wiki/Camera_matrix#:~:text=This%20type%20of%20camera%20matrix,as%20the%203D%20coordinate%20system.
    # https://answers.opencv.org/question/83807/normalized-camera-image-coordinates/
    # https://stackoverflow.com/questions/49571491/camera-calibration-reverse-projection-of-pixel-to-direction
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # https://web.archive.org/web/20150315234419/https://answers.opencv.org/question/4862/how-can-i-do-back-projection/


    # (-136.81, 72.52, 443.77)
    # (-144.71, -29.39, 446.14)


    # z = some value > 0
    # x = (u-cx)*z/fx
    # y = (v-cy)*z/fy 

def detect_saddle_points(image, show_possible_detections=False, show_aruco_detections=False, show_final_detections=False, undistort = False, optimize_camera_matrix=False, recrop=False):
    #print(image)
    frame = cv2.imread(image) 
    #frame = cv2.flip(frame, 1) # horizontal flip
    plt.figure(figsize=(20, 20))

    if undistort:
        print("Undistorting image...")
        frame = cv2.undistort(src = frame, cameraMatrix = camera_intrinsics_matrix, distCoeffs = camera_distortion_coefficients)
        #plt.imshow(frame, interpolation = "nearest")
        #plt.show()
        #new_filename = image.replace(".png","_undistorted.png")
        #cv2.imwrite(new_filename, frame)

    if optimize_camera_matrix:
        f_x, f_y, c_x, c_y, frame = undistortion_with_intrinsics(image, cropping=recrop)
        camera_intrinsics_matrix[0][0] = f_x
        camera_intrinsics_matrix[1][1] = f_y
        camera_intrinsics_matrix[0][2] = c_x
        camera_intrinsics_matrix[1][2] = c_y

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco_dict_selection) #aruco.DICT_ARUCO_ORIGINAL
    parameters =  aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 100
    parameters.adaptiveThreshWinSizeStep = 1
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000000, 0.001)


    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize = (5,5), zeroZone = (3,3), criteria = criteria)

    if len(corners) == 0:
        print("Failure to find ARUCO markers")
        return None, None, None, None

    if show_aruco_detections:
        if undistort:
            rotation_vectors, translation_vectors, board_relative_position_vectors = aruco.estimatePoseSingleMarkers(corners, size_of_marker, camera_intrinsics_matrix, camera_distortion_coefficients)
        else:
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
        #print("Found {} corners".format(corner_count)) 

        cleaned_corner_coordinates = []
        cleaned_ids = []
        for corner_coordinate, corner_id in zip(corner_coordinates, corner_ids):
            if corner_id not in dead_corners:
                cleaned_corner_coordinates.append(corner_coordinate)
                cleaned_ids.append(corner_id)

        #print("Before removing dead corners, {} corners found, and after {} corners".format(len(corner_ids), len(cleaned_ids)))
        corner_ids = np.array(cleaned_ids)
        corner_coordinates = np.array(cleaned_corner_coordinates)

        if corner_ids is not None:
            charuco_corners = corner_coordinates[:, 0]
            charuco_ids = corner_ids[:, 0]

        if charuco_ids.size > 0:
            rect_size = 5
            id_font = cv2.FONT_HERSHEY_DUPLEX
            id_scale = 2
            id_color = (0, 255, 0)
            rect_thickness = 1

            for (corner, corner_id) in zip(charuco_corners, charuco_ids):
                corner_x = int(corner[0])
                corner_y = int(corner[1])
                id_text = "{}".format(str(corner_id))
                id_coord = (corner_x + 2*rect_size, corner_y + 2*rect_size)
                cv2.rectangle(frame, (corner_x - rect_size, corner_y - rect_size),
                            (corner_x + rect_size, corner_y + rect_size),
                            id_color, thickness=rect_thickness)
                cv2.putText(frame, id_text, id_coord, id_font, id_scale, id_color)
                #print("(Corner {}): (u,v)=({},{})".format(corner_id, corner[0],corner[1]))
                #aruco.drawDetectedCornersCharuco(frame.copy(), corners, ids)

        # print(corner_ids)
        # print(corner_coordinates)

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

        return corner_coordinates, corner_ids, camera_intrinsics_matrix
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
    image = "/home/sense/3cobot/convergence_rgb.png"
    f_x, f_y, c_x, c_y = undistortion_with_intrinsics(image)
    image = image.replace(".png","_undistorted_with_new_intrinsics.png")
    corner_coordinates, corner_ids, _, _ = detect_saddle_points(image)
    print("Corner coordinates: {}".format(corner_coordinates))
    print("Corner IDs: {}".format(corner_ids))
    u = corner_coordinates[0][0][0]
    v = corner_coordinates[0][0][1]
    x,y,z = xyz_pixel_forward_projection_from_focal_point(u=u,v=v,f_x=f_x,f_y=f_y,c_x=c_x,c_y=c_y,z=100)


  #   from scanning import *

  #   structured_scan_processing = False
  #   if structured_scan_processing:
  #       working_directory = "/home/sense/3cobot"
  #       project_name = "sfm_v2_"
  #       scan_data = {}

  #       min_scan_index = 1
  #       max_image_index = 0
  #       max_cloud_index = 0

  #       png_images = [f for f in os.listdir(working_directory) if project_name in f and ".png" in f]
  #       for image in png_images:
  #           i = int(image.split(project_name)[1].split("view")[1].split("_")[0])
  #           scan_data["{}_image".format(i)] = image
  #           #print("({}) {}".format(i, image))
  #           x = float(image.split("_x")[1].split("_y")[0])
  #           y = float(image.split("_y")[1].split("_z")[0])
  #           z = float(image.split("_z")[1].split("_yaw")[0])
  #           yaw = float(image.split("_yaw")[1].split("_pitch")[0])
  #           pitch = float(image.split("_pitch")[1].split("_rgb")[0])
  #           #print("({}) (x,y,z,pitch,yaw)=({:.3f},{:.3f},{:.3f},{:.3f},{:.3f})".format(i,x,y,z,pitch,yaw))
  #           scan_data["{}_x".format(i)] = x
  #           scan_data["{}_y".format(i)] = y
  #           scan_data["{}_z".format(i)] = z
  #           scan_data["{}_pitch".format(i)] = pitch
  #           scan_data["{}_yaw".format(i)] = yaw
  #           if i > max_image_index:
  #               max_image_index = i

  #       ply_clouds = [f for f in os.listdir(working_directory) if project_name in f and ".ply" in f and "union" not in f]
  #       for cloud in ply_clouds:
  #           i = int(cloud.split(project_name)[1].split("view")[1].split("_")[0])
  #           scan_data["{}_cloud".format(i)] = cloud
  #           #print("({}) {}".format(i, cloud))
  #           if i > max_cloud_index:
  #               max_cloud_index = i

  #       if max_image_index <= max_cloud_index:
  #           max_index = max_image_index
  #       else:
  #           max_index = max_cloud_index

  #       print()
  #       print()

  #       all_images = []
  #       all_clouds = []

  #       for i in range(min_scan_index, max_index+1):
  #           all_images.append(scan_data["{}_image".format(i)])
  #           all_clouds.append(scan_data["{}_cloud".format(i)])

  #           print(scan_data["{}_image".format(i)])
  #           #print(scan_data["{}_cloud".format(i)])


  #       sample_size = 100
  #       images = all_images[:sample_size]
  #       cloud_names = all_clouds[:sample_size]

  #       use_precomputed_translations_rotations = False
  #       camera_calibrate_charuco = False
  #       calibrate_stereo = False

  #       # 1 then 2
  #       detect_charuco_only = True
  #       parse_sfm_localizations = False

  #       if detect_charuco_only:
  #            # Key data structures to enable
  #            # point_1 : x1 y1 x2 y2 ... xTOTAL_SEEN_IMAGES yTOTAL_SEEN_IMAGES\n
  #            # point_2 : x1 y1 x2 y2 ... xTOTAL_SEEN_IMAGES yTOTAL_SEEN_IMAGES\n
  #            # ...
  #            # point_N : x1 y1 x2 y2 ... xTOTAL_SEEN_IMAGES yTOTAL_SEEN_IMAGES\n

  #           max_point_index = 0
  #           point_index_map = {} # from raw board-based identifier to our own global identifier
  #           points_in_frames = [[] for image in images] # list of lists, where each inner list is a normalized point index
  #           normalized_point_to_x_per_image = [{} for image in images]
  #           normalized_point_to_y_per_image = [{} for image in images]

  #           print("Finding ChAruCo poses for images")
  #           for image_number, image in enumerate(images):
  #               print("Image {}".format(image_number))
  #               corner_coordinates, corner_ids, _, _ = detect_saddle_points(image)
  #               print(corner_coordinates)
  #               print()
  #               print(corner_ids)
  #               for corner_id, corner_coordinate in zip(corner_ids, corner_coordinates):
  #                   corner_id = int(corner_id[0])
  #                   x = float(corner_coordinate[0][0])
  #                   y = float(corner_coordinate[0][1])
  #                   normalized_id = point_index_map.get(corner_id, max_point_index)
  #                   if normalized_id == max_point_index:
  #                       point_index_map[corner_id] = max_point_index
  #                       max_point_index += 1
  #                       # update points_in_frames for previous images to -1,-1
  #                   print("(NORM ID={} ({})) := (x,y)=({},{})".format(normalized_id, corner_id, x, y))
  #                   normalized_point_to_x_per_image[image_number][normalized_id] = x
  #                   normalized_point_to_y_per_image[image_number][normalized_id] = y
  #                   points_in_frames[image_number].append(normalized_id)

  #           with open("{}/{}localizations.txt".format(working_directory,project_name), "w") as output_file:
  #               for point_index in range(max_point_index):
  #                   output_line = ""
  #                   for image_number, image in enumerate(images):
  #                       if point_index in points_in_frames[image_number]:
  #                           x = normalized_point_to_x_per_image[image_number][point_index]
  #                           y = normalized_point_to_y_per_image[image_number][point_index]
  #                       else:
  #                           x = "-1.00"
  #                           y = "-1.00"
  #                       output_line = ("{} {} {}").format(output_line, x, y)
  #                   output_line.lstrip()
  #                   if point_index == max_point_index - 1:
  #                       output_line = ("{}").format(output_line)
  #                   else:
  #                       output_line = ("{}\n").format(output_line)
  #                   output_file.write(output_line)

  #           print(normalized_point_to_x_per_image)
  #           print(normalized_point_to_x_per_image)
  #           print(points_in_frames)
  #           print(max_point_index)
  #           # Currently you are trying to save a global frame mapping where -1 -1 shows up for normalized values that do not appear in this image


  #           # translations = []
  #           #     for raw_translation in raw_translations:
  #           #         listed_translation = np.asarray(raw_translation).tolist()
  #           #         listed_translation = np.asarray([[translation[0] * 1000] for translation in listed_translation])
  #           #         translations.append(listed_translation)

  #           #     with open('{}rotations.npy'.format(project_name), 'wb') as output_file:
  #           #         np.save(output_file, rotations)
  #           #     with open('{}translations.npy'.format(project_name), 'wb') as output_file:
  #           #         np.save(output_file, translations)


  #       elif camera_calibrate_charuco:
  #           if use_precomputed_translations_rotations:
  #               with open('{}rotations.npy'.format(project_name), 'rb') as input_file:
  #                   rotations = np.load(input_file)
  #               with open('{}translations.npy'.format(project_name), 'rb') as input_file:
  #                   translations = np.load(input_file)
  #           else:
  #               print("Finding ChAruCo poses for images")
  #               rotations, raw_translations = find_charuco_poses(images)
  #               translations = []
  #               for raw_translation in raw_translations:
  #                   listed_translation = np.asarray(raw_translation).tolist()
  #                   listed_translation = np.asarray([[translation[0] * 1000] for translation in listed_translation])
  #                   translations.append(listed_translation)

  #               with open('{}rotations.npy'.format(project_name), 'wb') as output_file:
  #                   np.save(output_file, rotations)
  #               with open('{}translations.npy'.format(project_name), 'wb') as output_file:
  #                   np.save(output_file, translations)
  #       elif calibrate_stereo:
  #           if use_precomputed_translations_rotations:
  #               pass
  #               # with open('{}rotations.npy'.format(project_name), 'rb') as input_file:
  #               #     rotations = np.load(input_file)
  #               # with open('{}translations.npy'.format(project_name), 'rb') as input_file:
  #               #     translations = np.load(input_file)
  #           else:
  #               print("Finding rotation and transformation between two camera poses")
  #               pass
  #               # rotations, raw_translations = find_charuco_poses(images)
  #               # translations = []
  #               # for raw_translation in raw_translations:
  #               #     listed_translation = np.asarray(raw_translation).tolist()
  #               #     listed_translation = [[translation[0] * 1000] for translation in listed_translation]
  #               #     translations.append(listed_translation)

  #               # with open('{}rotations.npy'.format(project_name), 'wb') as output_file:
  #               #     np.save(output_file, rotations)
  #               # with open('{}translations.npy'.format(project_name), 'wb') as output_file:
  #               #     np.save(output_file, translations)       
  #       elif parse_sfm_localizations:
  #           processed_clouds = []
  #           for camera_number in range(sample_size):
  #               camera_rotations = cv2.FileStorage("/home/sense/camera_rotations.xml", cv2.FILE_STORAGE_READ)
  #               rotations = camera_rotations.getNode("camera_rotations").at(camera_number).mat()

  #               camera_translations = cv2.FileStorage("/home/sense/camera_translations.xml", cv2.FILE_STORAGE_READ)
  #               translations = camera_translations.getNode("camera_translations").at(camera_number).mat()

  #               print("Rotations:")
  #               print(rotations)
  #               print()
  #               print("Translations")
  #               print(translations)

  #               print("Loading point cloud...")
  #               cloud_name = cloud_names[camera_number]
  #               cloud = PointCloud(filename="/home/sense/3cobot/{}".format(cloud_name))
  #               cloud.downsample(reduce_by_1_divided_by_n=50)
  #               print("Transforming points")
  #               cloud.homogeneous_transform(translation_matrix=translations, rotation_matrix=rotations)
  #               #cloud.homogeneous_transform_inverted(translation_vector=translations, rotation_matrix=rotations)
  #               processed_clouds.append(cloud)

  #           print("Combining point clouds into single one")
  #           master_cloud = processed_clouds[0]
  #           for cloud in processed_clouds[1:]:
  #               master_cloud.add_point_cloud(cloud)

  #           print("Exporting to CSV")
  #           master_cloud.export_to_csv("{}union".format(project_name))
  #       else:
  #           print("No calibration technique specified")
  #           sys.exit(0)

  #   else:
  #       images = ["/home/sense/3cobot/analysis/3d_hdr_v2.png", "/home/sense/3cobot/analysis/3d_hdr_v2_rgb.png"]
  #       for image in images:
  #           print(image)
  #           corner_coordinates, corner_ids, _, _ = detect_saddle_points(image)
  #           print(corner_ids)
  #           print(corner_coordinates)


  # 


        # global_points 
        # with open(ground_truth_coordinates_file_name, "r") as input_file:



        #   mapped_normalized_ids.append(corner_id)
        #   x_normalized_image_coord, y_normalized_image_coord, z_normalized_image_coord = xyz_pixel_forward_projection_from_focal_point(u=u,v=v,f_x=f_x,f_y=f_y,c_x=c_x,c_y=c_y,z=1)

        #   x,y,z = plane.find_point_position_on_plane(x_normalized_image_coord, y_normalized_image_coord, z_normalized_image_coord, camera_extrinsics_matrix=camera_extrinsics_matrix)
        #   #points.add_points(x=x, y=y, z=z, red=255, green=0, blue=0)
        #   #points.add_superpoint(x=x, y=y, z=z, red=48, green=226, blue=229, sphere_radius=2, superpoint_samples=500)

        #   if multi_batch_recalibrate:
        #       if standard_deviation_history_per_corner_id.get(corner_id[0], False):
        #           standard_deviation_history_per_corner_id[corner_id[0]]["x"].append(x)
        #           standard_deviation_history_per_corner_id[corner_id[0]]["y"].append(y)
        #           standard_deviation_history_per_corner_id[corner_id[0]]["z"].append(z)
        #       else:
        #           standard_deviation_history_per_corner_id[corner_id[0]] = {"x": [x], "y": [y], "z": [z]}

        #   print("Normalized coordinate projection (x,y,z) = ({},{},{})".format(x,y,z))
        #   normalized_projected_coordinates.append([x,y,z])

        # visualization = "{}_v2.ply".format(project_name)
        # points.save_as_ply(visualization)

        # print("\n\nResults for interpolation between known points...")
        # distances = []
        # deviations = []
        # for coordinate_a, corner_id_a in zip(projected_coordinates, mapped_ids):
        #   for coordinate_b, corner_id_b in zip(projected_coordinates.copy(), mapped_ids.copy()):
        #       if corner_id_a - corner_id_b == 1 or corner_id_a - corner_id_b == 14:
        #           x1 = coordinate_a[0] 
        #           y1 = coordinate_a[1]
        #           z1 = coordinate_a[2]
        #           x2 = coordinate_b[0]
        #           y2 = coordinate_b[1]
        #           z2 = coordinate_b[2]
        #           distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        #           print("Distance between corner {} and corner {} = {:.3f}mm".format(corner_id_a, corner_id_b, distance))
        #           distances.append(distance)
        #           deviations.append(abs(32.000 - distance))

        # ground_truth_coordinates_file_name = "{}_3d_coordinates.csv".format(project_name)
        # print("\n\nSaving ground truth (x,y,z) coordinates extracted for this scan as {}".format(ground_truth_coordinates_file_name))
        # with open(ground_truth_coordinates_file_name, "w") as output_file:
        #   for coordinate, corner_id in zip(projected_coordinates, mapped_ids): 
        #           x = coordinate[0]
        #           y = coordinate[1]
        #           z = coordinate[2]
        #           output_file.write("{},{:.3f},{:.3f},{:.3f}\n".format(int(corner_id[0]), float(x), float(y), float(z)))

        # if len(deviations) > 0:
        #   average_deviation = sum(deviations) / len(deviations)
        #   print("\n\nAverage deviation of projected points: {:.3f}mm".format(average_deviation))

        # if exit_after_one:
        #   print("Exiting after one")
        #   sys.exit(0)



    ##### !!!



        # NOT YET VALID: PROJECT FROM PLANE OF CAMERA OUTWARD USING KNOWN RAYCAST ANGLES
        # print("\n\nResults for projecting from normalized coordinates...")
        # normalized_distances = []
        # normalized_deviations = []
        # for coordinate_a, corner_id_a in zip(normalized_projected_coordinates, mapped_normalized_ids):
        #   for coordinate_b, corner_id_b in zip(normalized_projected_coordinates.copy(), mapped_normalized_ids.copy()):
        #       if corner_id_a - corner_id_b == 1 or corner_id_a - corner_id_b == 26:
        #           x1 = coordinate_a[0]
        #           y1 = coordinate_a[1]
        #           z1 = coordinate_a[2]
        #           x2 = coordinate_b[0]
        #           y2 = coordinate_b[1]
        #           z2 = coordinate_b[2]
        #           distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        #           print("Distance between corner {} and corner {} = {:.3f}mm".format(corner_id_a, corner_id_b, distance))
        #           normalized_distances.append(distance)
        #           normalized_deviations.append(abs(40.000 - distance))


        # average_normalized_deviation = sum(normalized_deviations) / len(normalized_deviations)
        # #print("Average distance: {:.3f}mm".format(average_distance))
        # print("Average deviation: {:.3f}mm".format(average_normalized_deviation))
