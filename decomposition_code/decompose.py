#!/usr/bin/env python3

import subprocess
import cv2

def decompose_image(imgname, image_dir, save_G_dir, save_J_dir):
    run_matlab = f"matlab -nodisplay -nojvm -nosplash -nodesktop -r \"try demo_decomposition('{imgname}', '{image_dir}', '{save_G_dir}', '{save_J_dir}'); catch; end; quit;\" | tail -n +11"
    subprocess.run(run_matlab, shell=True, check=True)

# decompose_image("DSC00637", "./light-effects/")