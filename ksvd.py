
#### ksvd
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
import spectral
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ApproximateKSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements
        max_iter:
            Maximum number of iterations
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            I = gamma[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self, X):
        if min(X.shape) < self.n_components:
            D = np.random.randn(self.n_components, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, X):
        gram = D.dot(D.T)

        Xy = np.float32(D).dot(np.float32(X.T))

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._initialize(X)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = np.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)



def draw_all_wav(spec_image, all_wav_dir = "input_data/COMBINED/2/all_wav/"):

    for i in range (spec_image.shape[2]):
        img = spec_image[:,:,i]
        rgb = np.zeros((img.shape[0], img.shape[1], 3))
        rgb[:, :, 0] = img
        rgb[:, :, 1] = img
        rgb[:, :, 2] = img
        cv2.imwrite(all_wav_dir + "channel_" + str(i) + ".png", np.int64(rgb * 255))

    return


def remove_noisy(spec_image, s_v):
    if s_v == "SWIR":
        spec_image[:, :, 78:86] = 0
        spec_image[:, :, 154:270] = 0

    if s_v == "COMBINED":
        spec_image[:, :, 0:272] = 0
        spec_image[:, :, 274] = 0
        spec_image[:, :, 276] = 0
        spec_image[:, :, 477] = 0
        spec_image[:, :, 481] = 0
        spec_image[:, :, 484] = 0
        spec_image[:, :, 488] = 0
        spec_image[:, :, 492] = 0
        spec_image[:, :, 495] = 0
        spec_image[:, :, 499] = 0
        spec_image[:, :, 502] = 0
        spec_image[:, :, 505] = 0
        spec_image[:, :, 507] = 0
        spec_image[:, :, 501] = 0
        spec_image[:, :, 509] = 0
        spec_image[:, :, 513] = 0
        spec_image[:, :, 515] = 0
        spec_image[:, :, 517] = 0
        spec_image[:, :, 520] = 0
        spec_image[:, :, 541] = 0

    return spec_image


def draw_anomalies_on_img (norms_binary, name_hdr, result_dir):

    spec = spectral.envi.open(name_hdr)
    spec_image = np.array(spec.asarray())
    img = spec_image[:, :, 289]
    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:, :, 0] = img
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img
    rgb[norms_binary == 1] = [1,0,0]
    cv2.imwrite(result_dir + "/" + "anomalies" + ".png", np.int64(rgb * 255))

    return


def calc_NDVI_mat(name_hdr, result_dir, L=1):
    spec = spectral.envi.open(name_hdr)
    spec_image = np.array(spec.asarray())
    meta = spec.metadata
    c_850 = np.median(spec_image[:, :, 458:468], axis=2)
    c_750 = np.median(spec_image[:, :, 400:410], axis=2)
    ndvi = (c_850 - c_750) / (c_850 + c_750)
    savi = (c_850 - c_750) / (c_850 + c_750 + L) * (1 + L)
    cv2.imwrite(result_dir + "//" + "NDVI.png", np.int64(ndvi * 255))
    cv2.imwrite(result_dir + "//" + "SAVI.png", np.int64(savi * 255))

    ndvi[ndvi >= 0.2] = 1
    savi[savi >= 0.1] = 1

    cv2.imwrite(result_dir + "//" + "NDVI2.png", np.int64(ndvi * 255))
    cv2.imwrite(result_dir + "//" + "SAVI2.png", np.int64(savi * 255))

    img = spec_image[:,:,462]
    img[ndvi == 1] = 1
    img[savi == 1] = 1

    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:, :, 0] = img
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img
    cv2.imwrite(result_dir + "/" + "NDVISAVI3" + ".png", np.int64(rgb * 255))

    return ndvi, savi


def calc_pca(original_data, data, result_dir, n_components=3):

    # Standardize the data (mean centering)
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_standardized)

    # Reconstruct the data
    reconstructed_data = pca.inverse_transform(principal_components)

    # Calculate residuals (reconstruction errors)
    residuals = np.sum((data_standardized - reconstructed_data) ** 2, axis=1)
    threshold = np.percentile(residuals, 95)

    # Identify anomalies based on residuals
    residuals = residuals.reshape((original_data.shape[0], original_data.shape[1]))

    residuals[residuals < threshold] = 0
    residuals[residuals > threshold] = 1

    cv2.imwrite(result_dir + "//" + "PCA.png", np.int64(residuals * 255))

    img = original_data[:, :, 462]
    img[residuals == 1] = 1

    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:, :, 0] = img
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img
    cv2.imwrite(result_dir + "/" + "PCA3" + ".png", np.int64(rgb * 255))

    return


def calc_ndsi(name_hdr, result_dir, ndsi_threshold=0.25):
    spec = spectral.envi.open(name_hdr)
    spec_image = np.array(spec.asarray())
    meta = spec.metadata
    img = spec_image[:, :, 462]
    c_850 = np.mean(spec_image[:, :, 458:468], axis=2)
    c_600 = np.mean(spec_image[:, :, 300:345], axis=2)
    ndsi = (c_600 - c_850) / (c_850 + c_600)
    ndsi = np.abs(ndsi)
    ndsi[ndsi > ndsi_threshold] = 1
    cv2.imwrite(result_dir + "//" + "ndsi.png", np.int64(ndsi * 255))

    img[ndsi == 1] = 1
    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:, :, 0] = img
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img
    cv2.imwrite(result_dir + "/" + "ndsi3" + ".png", np.int64(rgb * 255))

    return ndsi


def anomaly_detection(folder_hdr, s_v):
    import os
    ls_dir = os.listdir(folder_hdr)
    result_dir = folder_hdr + "//result_anomaly"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    for file in ls_dir:
        if not file.endswith(".hdr"):
            continue

        name_hdr = folder_hdr + "//" + file
        spec = spectral.envi.open(name_hdr)
        if s_v == "COMBINED":
           ndvi, savi = calc_NDVI_mat(name_hdr, folder_hdr)
           ndsi = calc_ndsi(name_hdr, result_dir)

        spec_image = np.array(spec.asarray())
        spec_image = remove_noisy(spec_image,s_v)
        draw_all_wav(spec_image)
        pixel_array = spec_image.reshape(spec_image.shape[0] * spec_image.shape[1], spec_image.shape[2])

        calc_pca(spec_image, pixel_array, result_dir)

        norms = np.linalg.norm(pixel_array, axis=1)
        norms = norms.reshape(norms.shape[0], 1)
        norms = np.tile(norms.shape[0], 1)
        pixel_array = np.multiply(pixel_array, 1 / norms)

        ind = np.random.randint(pixel_array.shape[0], size=5000)
        rows = spec_image.shape[0]
        cols = spec_image.shape[1]

        Y = pixel_array[ind]
        del spec
        Y = np.float64(Y)
        dico1 = ApproximateKSVD(n_components=7, transform_n_nonzero_coefs=3)
        dico1.fit(Y)
        del Y
        D = dico1.components_
        X = dico1.transform(pixel_array)
        m_class = np.float32(X) @ np.float32(D)
        del D
        del X
        del dico1
        rm_class = pixel_array - m_class
        del m_class
        del pixel_array
        norms = np.linalg.norm(rm_class, axis=1)

        percentile = np.percentile(norms.ravel(), 95)
        norms_binary = norms.copy()
        norms_binary[norms_binary < percentile] = 0
        norms_binary[norms_binary >= percentile] = 1
        norms_binary = norms_binary.reshape((rows, cols))


        if s_v == "COMBINED":
            norms_binary[ndvi == 1] = 0
            norms_binary[savi == 1] = 0
            norms_binary[ndsi == 1] = 0

        draw_anomalies_on_img(norms_binary, name_hdr, result_dir)

        cv2.imwrite(result_dir + "//" + file.split(".")[0] + ".png", np.int64(norms_binary*255))
        del rm_class
        spec = spectral.envi.open(name_hdr)
        spec_image = np.array(spec.asarray())
        spec_array = spec_image.copy()
        spec_array[norms_binary == 0] = np.zeros((spec_array.shape[2]))
        
        name_hdr = result_dir + "//" + file.split(".")[0] + "_result.hdr"
        spectral.envi.save_image(name_hdr, spec_array, dtype=np.float32, metadata=spec.metadata, interleave="bsq",
                                 ext="", force=True)





anomaly_detection("input_data/COMBINED/3", "COMBINED")
# anomaly_detection("input_data/SWIR/2", "SWIR")