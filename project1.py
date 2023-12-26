import numpy as np
from src.cp_exr import readEXR, writeEXR
import os
import cv2
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from src.cp_hw2 import writeHDR
from src.utils import Dataloader, g_solve, get_hdr, get_I_linear, get_hdr_stack_and_z

Fusing_Methods = ['linear_hdr', 'log_hdr']
Weight_Methods = ['uniform', 'tent', 'gaussian', 'photon']
Image_types = ['tiff', 'jpg']
Z_min = 0.05
Z_max = 0.95
lamb = 1

'''输出16张HDR图像至exp_result文件夹中'''
for image_type in Image_types:
    path = os.path.join(os.getcwd(), 'data\door_' + image_type)
    z, hdr_stack = get_hdr_stack_and_z(path)
    reference = hdr_stack[:, :, :, :].copy()
    for fusing_method in Fusing_Methods:
        for weight_method in Weight_Methods:
            if image_type == 'jpg':
                Z_min = 0.05
                Z_max = 0.95
                g = g_solve(z, l=lamb, z_min=Z_min, z_max=Z_max, method=weight_method)
                fig0 = plt.figure()
                ax0 = fig0.add_subplot(111)
                ax0.plot(np.linspace(0, 255, 256), g)
                fig0.savefig(os.path.join(os.getcwd(), 'exp_result',
                                          fusing_method + '_' + weight_method + '_' + 'Inverse_map_g'))
                I_linear = get_I_linear(hdr_stack, g)
                linear_HDR = get_hdr(reference, I_linear, method= fusing_method, w_method=weight_method,
                                     z_min=Z_min, z_max=Z_max, image_type=image_type)
                writeHDR(os.path.join(os.getcwd(), 'exp_result',
                                      image_type + '_' + fusing_method + '_' + weight_method + '.HDR'), linear_HDR)

            if image_type == 'tiff':
                Z_min = 0.01
                Z_max = 0.99
                linear_HDR = get_hdr(reference, hdr_stack, method=fusing_method, w_method=weight_method,
                                     z_min=Z_min, z_max=Z_max, image_type=image_type)
                writeHDR(os.path.join(os.getcwd(), 'exp_result',
                                      image_type + '_' + fusing_method + '_' + weight_method + '.HDR'), linear_HDR)

