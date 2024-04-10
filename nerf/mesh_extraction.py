import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import time
import sys
from pytorch3d.ops.knn import knn_points
import os
from os.path import exists

project_base_directory = "data/dragon_scale"
device = torch.device('cuda:0') 

views = [v for v in range(0,50,10)]

verbose = True

if verbose:
    print()

# we still would like to penalize for lack-of-contiguity 
max_face_normal_to_camera_angle_deviation = 60 # degrees from perpendicular, where 90 degrees means camera sees face head-on, 0 degrees is inclined ("glancing")
max_edge_length = 0.002 # meters, maximum length of a triangle edge before we toss out the triangle
min_y_value_plane_filter = -0.76 # meters, a plane that can remove noise outside of target box
threshold_of_max_xyz_difference = 0.00100 # meters, maximum Euclidian (X,Y,Z) difference between nearest neighbors in adjacent views
# max_difference_in_curvature_for_nearest_neighbors_across_views = 4.0 # empirical max difference between a Riemannian-inspired topologically-invariant curvature metric for two views

optimize_curvature = False

def reindex_triangles_by_pruning_unused_vertices(vertices_xyz, triangle_vertices):
    # load in our vertices and triangle indices into GPU
    vertices_xyz = vertices_xyz.to(device=device)
    triangle_vertices = triangle_vertices.to(device=device)

    # unique function in torch will give (a) the indices in the original vector that are unique, and (b) the mapping from the original vector to the new vector
    # this happens to be exactly what we need to grab only the vertices we actually use, and then reindex the triangle references with the new smaller set of vertices
    unique_vertices_in_triangles, reindexed_triangle_vertices = torch.unique(torch.flatten(triangle_vertices), sorted=True, return_inverse=True)

    # we just need to reshape back to triangle
    reindexed_triangle_vertices = torch.reshape(input=reindexed_triangle_vertices, shape=(-1,3))

    # grab our actual (x,y,z) values
    final_vertices_xyz = vertices_xyz[unique_vertices_in_triangles]

    return final_vertices_xyz, reindexed_triangle_vertices


def compute_normal_angle_deviation_from_perpendicular(a, b, c):
    # a, b, c are all (x,y,z) coordinates, the angle returned is between them ABC
    ba = a - b
    bc = c - b
    cosine_numerator = torch.sum(ba*bc, axis=1)
    cosine_denominator_1 = torch.linalg.norm(ba, axis=1)
    cosine_denominator_2 = torch.linalg.norm(bc, axis=1)
    cosine_angle = cosine_numerator / ((cosine_denominator_1 * cosine_denominator_2) + 0.0000001)
    angles = torch.arccos(cosine_angle)
    degree_angles = torch.rad2deg(angles)
    degrees_from_perpendicular = torch.sqrt((degree_angles - 90.0)**2)    
    # no_nansense_degrees = torch.nan_to_num(degree_angles, nan=90.0)

    return degrees_from_perpendicular


all_optimized_xyz = []
all_optimized_rgb = []
all_cam_xyz = []
for i,view in enumerate(views):
    xyz = torch.from_numpy(np.load("{}/per_view_optimized_xyz_for_view_{}.npy".format(project_base_directory, view))).to(device=device)
    rgb = torch.from_numpy(np.load("{}/per_view_optimized_rgb_for_view_{}.npy".format(project_base_directory, view))).to(device=device)
    cam_xyz = np.load("{}/cam_xyz_{}.npy".format(project_base_directory, view)) 

    if verbose:
        print("Loaded optimized {} (x,y,z), {} (r,g,b) for view {} with cam (x,y,z) = {}".format(xyz.shape[0], rgb.shape[0], i, cam_xyz))

    all_optimized_xyz.append(xyz)
    all_optimized_rgb.append(rgb)
    all_cam_xyz.append(cam_xyz)

vertices_xyz_across_views = [] # for each view, list of (x,y,z) vertices, where the index of each vertex is referenced by the triangle 
triangle_vertices_across_views = [] # for each view, lists of 3-tuple indices in vertices_across_views[view]
triangle_rgb_across_views = [] # for each view, for each *triangle* face, (r,g,b) values
xyz_vertices_to_triangle_index = [] # for each view, for each *triangle* face, (r,g,b) values

for i,view in enumerate(views):
    xyz = all_optimized_xyz[i]
    rgb = all_optimized_rgb[i]

    adjacent_pixel_indices = np.load("{}/adjacent_pixel_indices_for_view_{}.npy".format(project_base_directory, view))       
    indices_of_nonborder_pixels = np.load("{}/indices_of_nonborder_pixels_for_view_{}.npy".format(project_base_directory, view)) 

    if verbose:
        print("\nLoaded {} adjacent pixel indices, {} indices of nonborder pixels for view {}".format(adjacent_pixel_indices.shape, indices_of_nonborder_pixels.shape, view))

    nonborder_xyz = xyz[indices_of_nonborder_pixels]
    nonborder_rgb = rgb[indices_of_nonborder_pixels]

    if verbose:
        print("\nExtracted the nonborder (x,y,z) of size {} and (r,g,b) of size {}".format(nonborder_xyz.shape, nonborder_rgb.shape))

    adjacent_xyz_top = xyz[adjacent_pixel_indices[:,0]]
    adjacent_xyz_bot = xyz[adjacent_pixel_indices[:,1]]
    adjacent_xyz_lef = xyz[adjacent_pixel_indices[:,2]]
    adjacent_xyz_rig = xyz[adjacent_pixel_indices[:,3]]

    if verbose:
        print("\nGrabbed the adjacent top ({}), bottom ({}), left ({}), and right ({}) (x,y,z) values".format(adjacent_xyz_top.shape,
                                                                                                        adjacent_xyz_bot.shape,
                                                                                                        adjacent_xyz_lef.shape,
                                                                                                        adjacent_xyz_rig.shape))
    v1 = (adjacent_xyz_lef + adjacent_xyz_top) / 2.0
    v2 = (adjacent_xyz_top + nonborder_xyz) / 2.0
    v3 = (adjacent_xyz_top + adjacent_xyz_rig) / 2.0
    v4 = (adjacent_xyz_lef + nonborder_xyz) / 2.0
    v5 = nonborder_xyz
    v6 = (nonborder_xyz + adjacent_xyz_rig) / 2.0
    v7 = (adjacent_xyz_lef + adjacent_xyz_bot) / 2.0
    v8 = (nonborder_xyz + adjacent_xyz_bot) / 2.0
    v9 = (adjacent_xyz_bot + adjacent_xyz_rig) / 2.0

    if verbose:
        print("\nComputed 8 adjacent pixel values (v1, v2, ... ,v9) for each pixel, where v5 = pixel")
        print("For example, for first pixel with (x,y,z) := v5 ={}...".format(v5[0,:]))
        print("Top left corner of pixel := v1 ={}".format(v1[0,:]))
        print("Top of pixel := v2 = {}".format(v2[0,:]))
        print("Top right corner of pixel := v3 = {}".format(v3[0,:]))
        print("Left of pixel := v4 = {}".format(v4[0,:]))
        print("The center of the pixel := v5 = {}".format(v5[0,:]))
        print("Right of pixel := v6 = {}".format(v6[0,:]))
        print("Bottom left corner of pixel := v7 = {}".format(v7[0,:]))
        print("Bottom of pixel := v8 = {}".format(v8[0,:]))
        print("Bottom right corner of pixel := v9 = {}".format(v9[0,:]))

    all_vertices = torch.cat([v1,v2,v3,v4,v5,v6,v7,v8,v9])

    if verbose:
        print("\nCombined pixel vertices into one big list of all vertices of size {}".format(all_vertices.shape))

    unique_vertices, nonunique_to_unique_vertex_index = torch.unique(torch.round(all_vertices, decimals=6), return_inverse=True, dim=0, sorted=True)
    vertices_xyz_across_views.append(unique_vertices)

    if verbose:
        print("\nFor all of the vertices, derived to 6 decimals (i.e. 1 micrometer) {} of {} are actually unique".format(unique_vertices.shape, nonunique_to_unique_vertex_index.shape))

    start = 0
    end = v1.shape[0]
    v1_id = nonunique_to_unique_vertex_index[start:end]

    start = end
    end += v2.shape[0]
    v2_id = nonunique_to_unique_vertex_index[start:end]

    start = end
    end += v3.shape[0]
    v3_id = nonunique_to_unique_vertex_index[start:end]
    
    start = end
    end += v4.shape[0]
    v4_id = nonunique_to_unique_vertex_index[start:end]
    
    start = end
    end += v5.shape[0]
    v5_id = nonunique_to_unique_vertex_index[start:end]
    
    start = end
    end += v6.shape[0]
    v6_id = nonunique_to_unique_vertex_index[start:end]
    
    start = end
    end += v7.shape[0]
    v7_id = nonunique_to_unique_vertex_index[start:end]
    
    start = end
    end += v8.shape[0]
    v8_id = nonunique_to_unique_vertex_index[start:end]
    
    start = end
    end += v9.shape[0]
    v9_id = nonunique_to_unique_vertex_index[start:end]
    
    if verbose:
        print("\nFor every one of the pixel border points (v1, v2, ..., v9), identified what its unique global index is in our new list of unique vertices")
        print("For example, for v2, for the first 5 points, we have that their unique global IDs are {}".format(v2_id[:5]))

    # counter-clockwise ordering for implied normal estimation!
    t1 = torch.transpose(torch.vstack([v4_id, v5_id, v1_id]), 0, 1)
    t2 = torch.transpose(torch.vstack([v5_id, v2_id, v1_id]), 0, 1)
    t3 = torch.transpose(torch.vstack([v5_id, v6_id, v2_id]), 0, 1)
    t4 = torch.transpose(torch.vstack([v6_id, v3_id, v2_id]), 0, 1)
    t5 = torch.transpose(torch.vstack([v7_id, v8_id, v4_id]), 0, 1)
    t6 = torch.transpose(torch.vstack([v8_id, v5_id, v4_id]), 0, 1)
    t7 = torch.transpose(torch.vstack([v8_id, v9_id, v5_id]), 0, 1)
    t8 = torch.transpose(torch.vstack([v9_id, v6_id, v5_id]), 0, 1)

    all_triangles = torch.cat([t1,t2,t3,t4,t5,t6,t7,t8])
    all_triangle_rgb = torch.cat([nonborder_rgb, nonborder_rgb, nonborder_rgb, nonborder_rgb, nonborder_rgb, nonborder_rgb, nonborder_rgb, nonborder_rgb])

    if verbose:
        print("\nAcquired a list of all possible triangles of size {}, referencing each of the unique global vertex indices".format(all_triangles.shape))

    p1 = unique_vertices[all_triangles[:,0]]
    p2 = unique_vertices[all_triangles[:,1]]
    p3 = unique_vertices[all_triangles[:,2]]

    if verbose:
        print("\nExtracted out for each triangle the (x,y,z) coordinates of each vertex, e.g. 1 of 3 vertices is of size {} with first (x,y,z) value {}".format(p1.shape, p1[0,:]))

    cam_xyz = torch.from_numpy(all_cam_xyz[i]).to(device=device)

    # filter by relative angle of face to camera
    p_1_2_normal = compute_normal_angle_deviation_from_perpendicular(a=p1, b=p2, c=cam_xyz)
    p_2_1_normal = compute_normal_angle_deviation_from_perpendicular(a=p2, b=p1, c=cam_xyz)
    p_2_3_normal = compute_normal_angle_deviation_from_perpendicular(a=p2, b=p3, c=cam_xyz)
    p_3_2_normal = compute_normal_angle_deviation_from_perpendicular(a=p3, b=p2, c=cam_xyz)
    p_3_1_normal = compute_normal_angle_deviation_from_perpendicular(a=p3, b=p1, c=cam_xyz)
    p_1_3_normal = compute_normal_angle_deviation_from_perpendicular(a=p1, b=p3, c=cam_xyz)

    if verbose:
        print("\nComputed normal angles of size {} for every edge, relative to the camera, e.g. for one edge we have an angle of {} degrees".format(p_1_2_normal.shape, p_1_2_normal[0]))

    normal_angle_to_camera = ((p_1_2_normal + p_2_1_normal + p_2_3_normal + p_3_2_normal + p_3_1_normal + p_1_3_normal) / 6.0).cpu().detach().numpy()

    if verbose:
        min_normal = np.min(normal_angle_to_camera)
        mean_normal = np.mean(normal_angle_to_camera)
        max_normal = np.max(normal_angle_to_camera)
        std_normal = np.std(normal_angle_to_camera)
        print("For the average of these normals, we have min={}, mean={}, max={}, std={} degrees".format(min_normal, mean_normal, max_normal, std_normal))

    good_normal_indices_0 = np.argwhere(normal_angle_to_camera < max_face_normal_to_camera_angle_deviation)[:,0]

    if verbose:
        print("\nAfter filtering out all normal angle above {}, we have {:,} of {:,} triangles that qualify".format(max_face_normal_to_camera_angle_deviation, good_normal_indices_0.shape[0], normal_angle_to_camera.shape[0]))

    # filter by distance of edge in triangle
    p1_p2_distances = torch.sqrt(torch.sum((p1 - p2)**2, dim=1))
    p2_p3_distances = torch.sqrt(torch.sum((p2 - p3)**2, dim=1)) 
    p3_p1_distances = torch.sqrt(torch.sum((p3 - p1)**2, dim=1))

    if verbose:
        print("\nThen we compute lengths of each of the edges {}, e.g. for the first p1-p2 edge, length is {} meters".format(p1_p2_distances.shape, p1_p2_distances[0]))

    good_edge_indices_1 = torch.argwhere(p1_p2_distances < max_edge_length)[:,0].cpu().detach().numpy()
    good_edge_indices_2 = torch.argwhere(p2_p3_distances < max_edge_length)[:,0].cpu().detach().numpy()
    good_edge_indices_3 = torch.argwhere(p3_p1_distances < max_edge_length)[:,0].cpu().detach().numpy()

    if verbose:
        print("\nFor each of the edges, we filter out edges greater than {} meters; for our 3 edges, we then have {:,} and {:,} and {:,} qualifying".format(max_edge_length, 
                                                                                                                                                good_edge_indices_1.shape[0],
                                                                                                                                                good_edge_indices_2.shape[0],
                                                                                                                                                good_edge_indices_3.shape[0]))

    good_position_indices_4 = torch.argwhere(p1[:,1] > min_y_value_plane_filter)[:,0].cpu().detach().numpy()
    good_position_indices_5 = torch.argwhere(p2[:,1] > min_y_value_plane_filter)[:,0].cpu().detach().numpy()   
    good_position_indices_6 = torch.argwhere(p3[:,1] > min_y_value_plane_filter)[:,0].cpu().detach().numpy()   

    if verbose:
        print("\nAs an example of a plane-based filtering, we remove all triangles with any points that have y < {}, which has {:,}, {:,}, {:,} qualifying points".format(min_y_value_plane_filter, 
                                                                                                                                                good_position_indices_4.shape[0],
                                                                                                                                                good_position_indices_5.shape[0],
                                                                                                                                                good_position_indices_6.shape[0]))


    # combine all of the good indices
    good_indices_0_1 = np.intersect1d(ar1=good_normal_indices_0, ar2=good_edge_indices_1)
    good_indices_0_1_2 = np.intersect1d(ar1=good_indices_0_1, ar2=good_edge_indices_2)
    good_indices_0_1_2_3 = np.intersect1d(ar1=good_indices_0_1_2, ar2=good_edge_indices_3)
    good_indices_0_1_2_3_4 = np.intersect1d(ar1=good_indices_0_1_2_3, ar2=good_position_indices_4)
    good_indices_0_1_2_3_4_5 = np.intersect1d(ar1=good_indices_0_1_2_3_4, ar2=good_position_indices_5)
    good_indices_0_1_2_3_4_5_6 = np.intersect1d(ar1=good_indices_0_1_2_3_4_5, ar2=good_position_indices_6)

    good_triangles = all_triangles[good_indices_0_1_2_3_4_5_6] #.cpu().detach().numpy()        
    good_triangle_rgb = all_triangle_rgb[good_indices_0_1_2_3_4_5_6].cpu().detach().numpy()
    # good_triangle_gradients = all_triangle_gradients[good_indices_0_1_2_3_4_5_6].cpu().detach().numpy()

    if verbose:
        print("\nThen we combine all of the above filters, to select only the {:,} of {:,} triangles with points and edges which satisfy *all* of the above constraints.".format(good_triangles.shape[0], 
                                                                                                                                                all_triangles.shape[0]))

    triangle_vertices_across_views.append(good_triangles)
    triangle_rgb_across_views.append(good_triangle_rgb)


# before proceeding with multi-view filtering, we prune the indices and vertices, for faster look-up
for i,view in enumerate(views):
    vertices_xyz = vertices_xyz_across_views[i]
    triangle_vertices = triangle_vertices_across_views[i]

    new_vertices_xyz, new_triangle_vertices = reindex_triangles_by_pruning_unused_vertices(vertices_xyz, triangle_vertices)
    if verbose:
        print("\nBefore reindexing view {}, we had {} vertices and {} triangles, after reindexing we have {} vertices and {} triangles".format(view,
                                                                                                                                             vertices_xyz.shape,
                                                                                                                                             triangle_vertices.shape,
                                                                                                                                             new_vertices_xyz.shape,
                                                                                                                                             new_triangle_vertices.shape,
                                                                                                                                             )) # new_triangle_vertices[0,:]

    vertices_xyz_across_views[i] = new_vertices_xyz
    triangle_vertices_across_views[i] = new_triangle_vertices

final_number_of_views = 0
points_remaining_to_fit = True
for i,view in enumerate(views):
    this_view = i
    comparisons = 2
    number_of_views = len(views) 
    comparison_views = []
    if not points_remaining_to_fit:
        break
    for comparison_view in [this_view-1,this_view+1]:
        if not points_remaining_to_fit:
            break
        if comparison_view == this_view:
            continue
        if comparison_view >= 0 and comparison_view <= number_of_views - 1:
            vertices_xyz = vertices_xyz_across_views[this_view]
            triangle_vertices = triangle_vertices_across_views[this_view]
            triangle_rgb = triangle_rgb_across_views[this_view]

            # lastly, filter out triangles that are too far away from their respective nearest neighbors in adjacent views
            if verbose:
                print("\nFor view {}, we compare multi-view geometry and color, and filter out points/triangles/colors which are too different from their KNN in adjacent views".format(view))

            # get (x,y,z) coordinates for other views
            nearby_xyz = vertices_xyz_across_views[comparison_view] #torch.cat([all_optimized_xyz[other_view] for other_view in comparison_views]) # (N_pixels * [V views - 1], 3)
            nearby_triangle_vertices = triangle_vertices_across_views[comparison_view] #torch.cat([all_optimized_xyz[other_view] for other_view in comparison_views]) # (N_pixels * [V views - 1], 3)

            total_number_of_other_points = nearby_xyz.shape[0]

            # computing nearest neighbor in nearest views for every point in proposed triangles
            xyz_distances, indices, nn = knn_points(p1=torch.unsqueeze(vertices_xyz, dim=0).to(device=device), p2=torch.unsqueeze(nearby_xyz, dim=0).to(device=device), K=1)

            nearest_neighbor_distances = xyz_distances[0,:,0]
            nearest_neighbor_indices = indices[0,:,0]

            p1_xyz_neighbor_distances = nearest_neighbor_distances[triangle_vertices[:,0]]
            p2_xyz_neighbor_distances = nearest_neighbor_distances[triangle_vertices[:,1]]
            p3_xyz_neighbor_distances = nearest_neighbor_distances[triangle_vertices[:,2]]

            if verbose:
                print("\nComparing view {} with {} points to view {} with {} points...".format(this_view, vertices_xyz.shape, comparison_view, nearby_xyz.shape))
                # print("For example, first value p1 distances are {}".format(p1_xyz_neighbor_distances[:5]))

            # get indices for triangle points with nearest neighbors that have (X,Y,Z) values sufficiently close
            p1_with_close_xyz_neighbors_0 = torch.argwhere(p1_xyz_neighbor_distances <= threshold_of_max_xyz_difference)[:,0].cpu().detach().numpy()
            p2_with_close_xyz_neighbors_1 = torch.argwhere(p2_xyz_neighbor_distances <= threshold_of_max_xyz_difference)[:,0].cpu().detach().numpy()
            p3_with_close_xyz_neighbors_2 = torch.argwhere(p3_xyz_neighbor_distances <= threshold_of_max_xyz_difference)[:,0].cpu().detach().numpy()

            if verbose:
                print("\nWe then filter out all points which have KNN with K=1 > {} meters away".format(threshold_of_max_xyz_difference)) 
                print("i.e. we keep {} of {} points for p1...".format(p1_with_close_xyz_neighbors_0.shape, p1_xyz_neighbor_distances.shape))
                print("i.e. we keep {} of {} points for p2...".format(p2_with_close_xyz_neighbors_1.shape, p2_xyz_neighbor_distances.shape))
                print("i.e. we keep {} of {} points for p3...".format(p3_with_close_xyz_neighbors_2.shape, p3_xyz_neighbor_distances.shape))


            if optimize_curvature:
                # get the 3 respective points for each triangle in this view
                p1 = vertices_xyz[triangle_vertices[:,0]] * 1000
                p2 = vertices_xyz[triangle_vertices[:,1]] * 1000
                p3 = vertices_xyz[triangle_vertices[:,2]] * 1000

                # triangle curvature gradients for this view
                this_view_triangle_x_deviation = torch.sqrt((p1[:,0] - p2[:,0])**2) + torch.sqrt((p2[:,0] - p3[:,0])**2) + torch.sqrt((p3[:,0] - p1[:,0])**2)
                this_view_triangle_y_deviation = torch.sqrt((p1[:,1] - p2[:,1])**2) + torch.sqrt((p2[:,1] - p3[:,1])**2) + torch.sqrt((p3[:,1] - p1[:,1])**2)
                this_view_triangle_z_deviation = torch.sqrt((p1[:,2] - p2[:,2])**2) + torch.sqrt((p2[:,2] - p3[:,2])**2) + torch.sqrt((p3[:,2] - p1[:,2])**2)
                this_view_riemann_metric = torch.stack([this_view_triangle_x_deviation, this_view_triangle_y_deviation, this_view_triangle_z_deviation], dim=1)            

                if verbose:
                    print("\nFor an invariant metric of curvature, we compute the sum for every edge of the triangle of the total deviation for each of x,y,z") 
                    print("That leads to for one dimension (x) a metric of size {} with first 5 values {}".format(this_view_triangle_x_deviation.shape, this_view_triangle_x_deviation[:5]))
                    print("The Riemann metric itself looks like {} with first 5 values: {}".format(this_view_riemann_metric.shape, this_view_riemann_metric[:5,:]))

                # get the 3 respective points for each triangle in nearby view
                nearby_p1 = nearby_xyz[nearby_triangle_vertices[:,0]] * 1000
                nearby_p2 = nearby_xyz[nearby_triangle_vertices[:,0]] * 1000
                nearby_p3 = nearby_xyz[nearby_triangle_vertices[:,0]] * 1000

                # triangle curvature gradients for this view
                other_view_triangle_x_deviation = torch.sqrt((nearby_p1[:,0] - nearby_p2[:,0])**2) + torch.sqrt((nearby_p2[:,0] - nearby_p3[:,0])**2) + torch.sqrt((nearby_p3[:,0] - nearby_p1[:,0])**2)
                other_view_triangle_y_deviation = torch.sqrt((nearby_p1[:,1] - nearby_p2[:,1])**2) + torch.sqrt((nearby_p2[:,1] - nearby_p3[:,1])**2) + torch.sqrt((nearby_p3[:,1] - nearby_p1[:,1])**2)
                other_view_triangle_z_deviation = torch.sqrt((nearby_p1[:,2] - nearby_p2[:,2])**2) + torch.sqrt((nearby_p2[:,2] - nearby_p3[:,2])**2) + torch.sqrt((nearby_p3[:,2] - nearby_p1[:,2])**2)
                other_view_riemann_metric = torch.stack([other_view_triangle_x_deviation, other_view_triangle_y_deviation, other_view_triangle_z_deviation], dim=1)            

                p1_to_other_p1_xyz_distances, indices, nn = knn_points(p1=torch.unsqueeze(p1, dim=0).to(device=device), p2=torch.unsqueeze(nearby_p1, dim=0).to(device=device), K=1)
                p1_to_other_p1_nearest_neighbor_indices = indices[0,:,0]

                if verbose:
                    print("\nTo evaluate curvature, comparing view {} p1 with {} points to view {} p1 with {} points...".format(this_view, p1.shape, comparison_view, nearby_p1.shape))
                    print("For example, first value p1 distances are {}".format(p1_to_other_p1_xyz_distances[:5]))

                this_view_curvature = this_view_riemann_metric
                nearest_neighbor_curvature = other_view_riemann_metric[p1_to_other_p1_nearest_neighbor_indices]

                difference_in_curvature = torch.sum((this_view_curvature - nearest_neighbor_curvature)**2, dim=1)
                if verbose:
                    print("\nFor curvature differences, we have {} with first 5 as {}".format(difference_in_curvature.shape, difference_in_curvature[:5]))
                    print("Min: {}, Mean: {}, Max: {}, Std: {}".format(torch.min(difference_in_curvature),
                                                                       torch.mean(difference_in_curvature),
                                                                       torch.max(difference_in_curvature),
                                                                       torch.std(difference_in_curvature)))


                p1_with_nearest_neighbors_close_in_curvature_energy = torch.argwhere(difference_in_curvature <= max_difference_in_curvature_for_nearest_neighbors_across_views)[:,0].cpu().detach().numpy()
                
                if verbose:
                    print("By filtering out triangles which are too distant in curvature metric, we now have {} of {} points for p1...".format(p1_with_nearest_neighbors_close_in_curvature_energy.shape, difference_in_curvature.shape))

            # combine all of the good indices
            good_indices_0_1 = np.intersect1d(ar1=p1_with_close_xyz_neighbors_0, ar2=p2_with_close_xyz_neighbors_1)
            good_indices_0_1_2 = np.intersect1d(ar1=good_indices_0_1, ar2=p3_with_close_xyz_neighbors_2)

            if optimize_curvature:
                good_indices_0_1_2_3 = np.intersect1d(ar1=good_indices_0_1_2, ar2=p1_with_nearest_neighbors_close_in_curvature_energy)
                good_triangle_vertices = triangle_vertices[good_indices_0_1_2_3]         
                good_triangle_rgb = triangle_rgb[good_indices_0_1_2_3] # update our lists of good triangles and their corresponding (R,G,B) values

                if verbose:
                    print("\nAfter taking only the triangles in which all points meet the KNN and curvature criteria above, for view {} we have {} points out of {}".format(view,
                                                                                                                                                                            good_triangle_vertices.shape,
                                                                                                                                                                            triangle_vertices.shape))                      
            else:
                good_triangle_vertices = triangle_vertices[good_indices_0_1_2]         
                good_triangle_rgb = triangle_rgb[good_indices_0_1_2]                    

                                                                                                                                     


            final_vertices_xyz, final_triangle_vertices = reindex_triangles_by_pruning_unused_vertices(vertices_xyz, good_triangle_vertices)

            if good_triangle_vertices.shape[0] > 0: 
                if verbose:
                    print("\nBefore last round of reindexing view {}, we had {} vertices and {} triangles (e.g. {}), after reindexing we have {} vertices and {} triangles (e.g. {})".format(view,
                                                                                                                                                                         vertices_xyz.shape,
                                                                                                                                                                         good_triangle_vertices.shape,
                                                                                                                                                                         good_triangle_vertices[0,:],
                                                                                                                                                                         final_vertices_xyz.shape,
                                                                                                                                                                         final_triangle_vertices.shape,
                                                                                                                                                                         final_triangle_vertices[0,:]))



            # and correspondingly update the global indices
            vertices_xyz_across_views[i] = final_vertices_xyz
            triangle_vertices_across_views[i] = final_triangle_vertices
            triangle_rgb_across_views[i] = good_triangle_rgb

            
            final_number_of_views = this_view
            if good_triangle_vertices.shape[0] == 0:
                print("Optimization ending early because no more points left to fit, and pruning. Consider relaxing requirements.")
                points_remaining_to_fit = False


for i,view in enumerate(range(final_number_of_views)):
    vertices = vertices_xyz_across_views[i].cpu().detach().numpy()
    triangles = triangle_vertices_across_views[i].cpu().detach().numpy()
    colors = (triangle_rgb_across_views[i] * 255).astype(int)

    if verbose:
        print("\nSaving final mesh .ply for view {} with {} vertices, {} triangles, {} colors".format(view,
                                                                                                      vertices.shape,
                                                                                                      triangles.shape,
                                                                                                      colors.shape)) 

    number_of_vertices = vertices.shape[0]
    number_of_triangles = triangles.shape[0]

    filename = "{}/mesh_{}.ply".format(project_base_directory, i)
    fout = open(filename, "w")
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex {}\n".format(str(number_of_vertices)))
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face {}\n".format(str(number_of_triangles)))
    fout.write("property list uchar int vertex_indices\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    fout.write("end_header\n")
    offsets = []

    for vertex_number in range(number_of_vertices):
        x = vertices[vertex_number, 0]
        y = vertices[vertex_number, 1]
        z = vertices[vertex_number, 2]
        vertex_string = "{} {} {}\n".format(x,y,z)
        fout.write(vertex_string)

    for i,tri in enumerate(triangles):
        col = colors[i,:]
        triangle_string = "3 {} {} {} {} {} {}\n".format(tri[0],tri[1],tri[2], col[0], col[1], col[2])
        fout.write(triangle_string)