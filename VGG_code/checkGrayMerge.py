#!/usr/bin/env python3

import subprocess

run_matlab = "matlab -nodisplay -nojvm -nosplash -nodesktop -r \"run('checkGrayMerge.m');exit;\" | tail -n +11"

subprocess.run(run_matlab, shell=True, check=True)


# img = cv2.imread("./results_VGGfeatures/DSC01607_input.jpg").astype(np.float32) / 255.0

# img_r = img[:, :, 0]
# img_g = img[:, :, 1]
# img_b = img[:, :, 2]


# def cat3(x):
#     return np.dstack((x, x, x))


# def norm(x):
#     max_x = np.max(x)
#     min_x = np.min(x)
#     return (x - min_x) / (max_x - min_x)


# # Crit-2 (using Merten's fusion approach)
# img_stack = np.zeros((img_r.shape[0], img_r.shape[1], 3, 3))
# img_stack[:, :, :, 0] = cat3(img_r)
# img_stack[:, :, :, 1] = cat3(img_g)
# img_stack[:, :, :, 2] = cat3(img_b)

# # gray merge only with saturation param
# img_gray_weights_2, img_gray_best2_2 = exposure_fusion(img_stack, [0, 1, 0])
# # cv2.imshow("Image - Merten's Fused Gray", np.hstack((img, img_gray_best2_2)))
# # cv2.waitKey(0)

# cv2.imwrite(f"./results_VGGfeatures/{imgname}_GrayBest.jpg", 255. * img_gray_best2_2)
# cv2.imwrite(f"./results_VGGfeatures/{imgname}_I_GrayBest.jpg", 255. * np.hstack((img, img_gray_best2_2)))