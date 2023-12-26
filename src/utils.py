import os
import numpy as np
import cv2

class Dataloader:
    def __init__(self, path, verbose=True):
        self.file = os.listdir(path)
        self.file.sort(key=lambda x: int(x.split('.')[0].split('e')[-1]))
        self._mat_list = [os.path.join(path, file) for file in self.file]
        self.verbose = verbose

    def __iter__(self):
        for file_name, file_path in zip(self.file, self._mat_list):
            # 刚读入时为(31, 1392, 1300) --> 转置为(1392, 1300, 31)以便resize
            HDR_tiff = cv2.imread(file_path, 3)
            h, w, c = HDR_tiff.shape
            if self.verbose:
                print("Decoding HDR dataset: <shape=", HDR_tiff.shape, ">@", "file_name:", file_name)
            # 用于数据增强，返回 len = 10 的列表，里面储存 (512, 512, 31) 的元素
            HDR_flat = HDR_tiff[h // 10::h // 10, w // 10::w // 10, :].reshape(-1, 1)
            yield np.float32(HDR_tiff), np.float32(HDR_flat)


def w(z, z_min, z_max, method='uniform', img_index=None):
    if method == 'uniform':
        if z_min <= z <= z_max:
            w = 1
        else:
            w = 0
    if method == 'tent':
        if z_min <= z <= z_max:
            w = np.minimum(z, 1 - z)
        else:
            w = 0
    if method == 'gaussian':
        if z_min <= z <= z_max:
            w = np.exp(-4*(z-0.5)**2/0.5**2)
        else:
            w = 0
    if method == 'photon':
        if z_min <= z <= z_max:
            w = 2 ** img_index / 2048
        else:
            w = 0
    return w


def w_vectorize(z, z_min, z_max, method='uniform'):
    if method == 'uniform':
        w = np.ones_like(z)
        w[z > z_max] = 0
        w[z < z_min] = 0
    if method == 'tent':
        w = np.minimum(z, 1-z)
        w[z > z_max] = 0
        w[z < z_min] = 0
    if method == 'gaussian':
        w = np.exp(-4*(z-0.5)**2/0.5**2)
        w[z > z_max] = 0
        w[z < z_min] = 0
    if method == 'photon':
        exposure_time = np.logspace(0, 15, 16, base=2) / 2048
        w = exposure_time.reshape(1, 1, 1, 16) * np.ones_like(z)
        w[z > z_max] = 0
        w[z < z_min] = 0
    return w


def g_solve(z, l=1, z_min=0.05, z_max=0.95, method='tent'):
    n = 256
    A = np.zeros((z.shape[0] * z.shape[1] + n - 2 + 1, n + z.shape[0]))
    b = np.zeros((z.shape[0] * z.shape[1] + n - 2 + 1, 1))

    k = 0
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            wij = w(z[i, j] / 255, z_min, z_max, method=method, img_index=j)
            A[k, int(z[i, j])] = wij
            A[k, n + i] = - wij
            b[k] = wij * np.log(2 ** j / 2048)
            k += 1

    A[k, 256 // 2] = 1

    k += 1

    for i in range(1, n - 1):
        if method == 'photon':
            wl = 1
        else:
            wl = w(i / 255, z_min, z_max, method=method)

        # wl_plus = w((i + 1) / 255, z_min, z_max, method=method)
        # wl_minus = w((i - 1) / 255, z_min, z_max, method=method)
        A[k, i] = -2 * l * wl
        A[k, i + 1] = l * wl
        A[k, i - 1] = l * wl
        k += 1
        # np.zeros((z.shape[0]*z.shape[1] + n + 1, n + z.shape[0]), dtype=np.float16)

    x = np.linalg.lstsq(A, b, rcond=None)

    g = x[0][0:256]
    return g


def get_hdr(reference, hdr_linear_stack, method='linear_hdr', w_method='tent' ,z_min=0.05, z_max=0.95, image_type='jpg'):
    if image_type == 'jpg':
        reference = reference / 255

    if image_type == 'tiff':
        reference = reference / (2**16-1)

    w_vector = w_vectorize(reference, z_min, z_max, method=w_method)
    exposure_time = np.logspace(0, 15, 16, base=2, dtype=np.float32) / 2048
    exposure_time = exposure_time.reshape(1, 1, 1, 16)

    if method == 'linear_hdr':
        denominator = np.sum(w_vector, axis=3, keepdims=False)
        numerator = np.sum(w_vector * hdr_linear_stack / exposure_time, axis=3, keepdims=False)
        HDR = numerator / (denominator+1e-16)
        # HDR[denominator<1e-8]

    if method == 'log_hdr':
        denominator = np.sum(w_vector, axis=3, keepdims=False)
        numerator = np.sum(w_vector * (np.log(hdr_linear_stack+1e-16) - np.log(exposure_time)), axis=3, keepdims=False)
        HDR = np.exp(numerator / (denominator + 1e-16))

    anormal = np.argwhere(denominator == 0)
    index1 = np.argwhere(reference[anormal[:, 0], anormal[:, 1], anormal[:, 2], 8] < 0.5)
    index2 = np.argwhere(reference[anormal[:, 0], anormal[:, 1], anormal[:, 2], 8] >= 0.5)
    index3 = anormal[index1[:, 0], :]
    index4 = anormal[index2[:, 0], :]
    if index3.shape[0] != 0:
        HDR[index3[:, 0], index3[:, 1], index3[:, 2]] = np.min(HDR[denominator != 0])
    if index4.shape[0] != 0:
        HDR[index4[:, 0], index4[:, 1], index4[:, 2]] = np.max(HDR[denominator != 0])
    return HDR


def get_I_linear(hdr_stack, g):
    for i in range(256):
        hdr_stack[hdr_stack == i] = g[i]
    I_linear = np.exp(hdr_stack)
    return I_linear


def get_hdr_stack_and_z(path):
    dataloader = Dataloader(path)

    for j, (hdr, img) in enumerate(dataloader):
        if j == 0:
            z = img
            hdr_stack = hdr.reshape((*hdr.shape, 1))
        else:
            z = np.concatenate((z, img), axis=1)
            hdr_stack = np.concatenate((hdr_stack, hdr.reshape((*hdr.shape, 1))), axis=3)
        print(img.shape)
    return z, hdr_stack