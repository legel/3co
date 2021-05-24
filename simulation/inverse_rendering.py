import numpy as np
import cv2
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow_graphics.rendering.reflectance import lambertian, phong
from tensorflow_graphics.geometry.representation import vector
from math import pi
import PIL
import os

number_of_iterations = 0

def render(xyz, normals, albedo, light_position, specular_percentage, shininess, camera_position):
  diffuse_percentage = 1 - specular_percentage

  # hack: we overwrite light_position, because for some reason using the light position = camera position does not work
  light_position = np.array((xyz.shape[0] / 2.0, xyz.shape[1] / 2.0, 0.0), dtype=np.float64)

  # pure white light, with normalized intensity based on distance to object 
  light_intensity_scale = vector.dot(light_position, light_position, axis=-1) * 4.0 * pi

  light_red, light_green, light_blue = [1, 1, 1]
  light_intensity = np.array((light_red, light_green, light_blue), dtype=np.float32) * light_intensity_scale 

  surface_normal = tf.math.l2_normalize(xyz - normals, axis=-1)

  incoming_light_direction = tf.math.l2_normalize(xyz - light_position, axis=-1)
  outgoing_light_direction = tf.math.l2_normalize(xyz - camera_position, axis=-1)

  brdf_lambertian = diffuse_percentage * lambertian.brdf(incoming_light_direction, outgoing_light_direction, surface_normal, albedo)
  brdf_phong = specular_percentage * phong.brdf(incoming_light_direction, 
                                                outgoing_light_direction, 
                                                surface_normal,
                                                np.array((shininess,), dtype=np.float64), 
                                                albedo)
  brdf_composite = brdf_lambertian + brdf_phong

  # Irradiance
  cosine_term = vector.dot(surface_normal, -incoming_light_direction)
  cosine_term = tf.math.maximum(tf.zeros_like(cosine_term), cosine_term)

  vector_light_to_surface = xyz - light_position

  light_to_surface_distance_squared = vector.dot(vector_light_to_surface, vector_light_to_surface, axis=-1)

  # prepare data structures so they match bit size
  light_intensity = tf.cast(light_intensity, tf.float64)
  light_to_surface_distance_squared = tf.cast(light_to_surface_distance_squared, tf.float64)
  cosine_term = tf.cast(cosine_term, tf.float64)

  # compute irradince
  irradiance = light_intensity / (4 * pi * light_to_surface_distance_squared) * cosine_term

  # Rendering equation
  zeros = tf.zeros(xyz.shape)
  radiance = brdf_composite * irradiance
  radiance_lambertian = brdf_lambertian * irradiance
  radiance_phong = brdf_phong * irradiance

  # Saturates radiances at 1 for rendering purposes.
  radiance = np.minimum(radiance, 1.0)
  radiance_lambertian = np.minimum(radiance_lambertian, 1.0)
  radiance_phong = np.minimum(radiance_phong, 1.0)

  # Gamma correction
  radiance = np.power(radiance, 1.0 / 2.2)
  radiance_lambertian = np.power(radiance_lambertian, 1.0 / 2.2)
  radiance_phong = np.power(radiance_phong, 1.0 / 2.2)

  return radiance


def save_image(image_data, output_file_path):
  image_data = (image_data * 255).astype(np.uint8)
  image_data = PIL.Image.fromarray(image_data)
  image_data.save(output_file_path) 


def read_geometry_from_exr(exr_image_path):
  exr_image = cv2.imread(exr_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  xyz_data = np.array(exr_image, dtype=np.float64)
  image_width, image_height, image_depth = xyz_data.shape
  xyz = np.zeros((image_width, image_height, image_depth), dtype=np.float64)
  xyz[:,:,0], xyz[:,:,1], xyz[:,:,2] = xyz_data[:,:,2], xyz_data[:,:,1], xyz_data[:,:,0]
  return xyz


def read_diffuse_colors_from_png(png_image_path):
  print(png_image_path)
  diffuse_color_image_input = cv2.imread(png_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  print(diffuse_color_image_input.shape)
  diffuse_color_image = cv2.cvtColor(diffuse_color_image_input, cv2.COLOR_BGR2RGB)
  rgb_data = np.array(diffuse_color_image, dtype=np.float64)
  image_width, image_height, image_depth = rgb_data.shape
  image_shape = np.zeros((image_width,image_height))
  diffuse_colors = cv2.normalize(rgb_data, image_shape, 0, 1, cv2.NORM_MINMAX)
  return diffuse_colors


def read_normals_from_exr(exr_image_path):
  exr_image = cv2.imread(exr_image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  normals_data = np.array(exr_image, dtype=np.float64)
  image_width, image_height, image_depth = normals_data.shape
  normals = np.zeros((image_width, image_height, image_depth), dtype=np.float64)
  normals[:,:,0], normals[:,:,1], normals[:,:,2] = normals_data[:,:,2], normals_data [:,:,1], normals_data[:,:,0]
  return normals


def render_and_save_output( xyz, 
                            normals,
                            diffuse_colors = np.asarray([0.248636, 0.504908, 0.2400332], dtype=np.float64), 
                            image_dimensions = (2048, 2048, 3),
                            camera_position = [0.8, 0.0, 0.0],
                            light_position = [0.8, 0.0, 0.0],
                            specular_percentage = 0.2,
                            shininess = 9,
                            output_file_path="outputs/scipy_sims/inverse_render_target.png"):

  if diffuse_colors.shape[0] == 3: # if only a single color, broadcast color to size of image
    diffuse_colors = tf.broadcast_to(diffuse_colors, xyz.shape)

  observation = render(xyz, normals, diffuse_colors, light_position, specular_percentage, shininess, camera_position)

  save_image(observation, output_file_path)
  return observation


def make_directory(output_directory_name):
  if not os.path.exists(output_directory_name):
    os.makedirs(output_directory_name)


def inverse_rendering_optimization_of_diffuse_colors(observation, xyz, normals, output_directory):

  def inverse_rendering_mean_squared_error(observation, hypothesis):
    observation = observation.flatten()
    hypothesis = hypothesis.flatten()
    loss = sum((observation - hypothesis)**2) / len(observation)
    return loss

  def inverse_render_evaluation(rgb_hypothesis, args):
    global number_of_iterations
    observation = args[0]
    xyz = args[1]
    normals = args[2]
    output_directory = args[3]

    # wrap up hypothesis
    r, g, b = rgb_hypothesis[0], rgb_hypothesis[1], rgb_hypothesis[2]
    hypothesis_albedo = np.array((r, g, b), dtype=np.float64)
    hypothesis_albedo = tf.broadcast_to(hypothesis_albedo, tf.shape(observation))

    output_file_path = "{}/{}.png".format(output_directory, number_of_iterations)

    # render with everything known but diffuse colors
    rendered_hypothesis = render_and_save_output( xyz = xyz, 
                                                  normals = normals, 
                                                  diffuse_colors = hypothesis_albedo, 
                                                  output_file_path = output_file_path)

    # save_image(image_data=rendered_hypothesis, output_file_path="{}/{}.png".format(output_directory, number_of_iterations))
    loss = inverse_rendering_mean_squared_error(observation, rendered_hypothesis)
    print('(LOSS: {}) R: {:.3f} G: {:.3f} B: {:.3f}'.format(loss, r, g, b))
    number_of_iterations += 1
    return loss

  final_result = minimize(  fun=inverse_render_evaluation, 
                            x0=[0.5, 0.5, 0.5],
                            options={'maxiter': 5, 'disp': True}, 
                            method='BFGS',
                            args=[observation, xyz, normals, output_directory])

  final_hypotheses = final_result.x
  return final_hypotheses[0], final_hypotheses[1], final_hypotheses[2]


if __name__ == "__main__":
  cwd = os.getcwd()

  geometry_exr_path = '{}/pillow_1_geometry.exr'.format(cwd)
  normals_exr_path = '{}/pillow_1_normal_output.exr'.format(cwd)
  diffuse_png_path = '{}/pillow_1_diffuse_colors.png'.format(cwd)

  xyz = read_geometry_from_exr(exr_image_path=geometry_exr_path)
  normals = read_normals_from_exr(exr_image_path=normals_exr_path)

  # missing "pillow_1_diffuse_colors.png", so unable to test the full multi-color render version
  # diffuse_colors = read_diffuse_colors_from_png(png_image_path=diffuse_png_path)

  output_directory = "{}/scipy_optimization_outputs".format(cwd)
  make_directory(output_directory)

  # hack for a single color optimization
  diffuse_colors = np.asarray([0.248636, 0.504908, 0.2400332], dtype=np.float64)

  observation = render_and_save_output( xyz=xyz,
                                        normals=normals,
                                        diffuse_colors=diffuse_colors,
                                        output_file_path="{}/observation.png".format(output_directory))

  inverse_rendering_optimization_of_diffuse_colors(observation, xyz, normals, output_directory)

