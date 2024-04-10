import numpy as np

matches_file = "/home/photon/sense/3cology/research/SuperGluePretrainedNetwork/example_output_directory/000200_000201_matches.npz"
matches_data = np.load(matches_file)

keypoints_image_0   = matches_data["keypoints0"]
keypoints_image_1   = matches_data["keypoints1"]
matches_from_0_to_1 = matches_data["matches"]
match_confidences   = matches_data["match_confidence"]

print("\nKeypoints for Image 0 (pixel locations, height and width, necessary for projecting out with camera intrinsics)\n{}: {}".format(keypoints_image_0.shape, keypoints_image_0))
print("\nKeypoints for Image 1\n{}: {}".format(keypoints_image_1.shape, keypoints_image_1))
print("\nMatches from Image 0 to Image 1 - \"For each keypoint in keypoints0, the matches array indicates the index of the matching keypoint in keypoints1, or -1 if the keypoint is unmatched.\"\n{}: {} (max: {})".format(matches_from_0_to_1.shape, matches_from_0_to_1,  np.max(matches_from_0_to_1)))
print("\nMatch Confidences\n{}: {}\n".format(match_confidences.shape, match_confidences))

