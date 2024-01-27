#### hyper registration

import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
import spectral
import cv2
import os
#import geopandas
# from shapely.geometry import Point
from scipy.spatial import KDTree, ConvexHull
import matplotlib.pyplot as plt
import pickle
import random
import sys
import os
import time
import tqdm
import SimpleITK as sitk



def read_images(folder_hdr, s_v, result_dir):
    '''
    Read image by hdr file
    
    '''   

    ls_dir = os.listdir(folder_hdr)
    for i,file in enumerate(ls_dir):
        if not file.endswith(".hdr"):
            continue

        name_hdr = folder_hdr + "//" + file 
        spec = spectral.envi.open(name_hdr)
        spec_image = np.array(spec.asarray())
       

        # if s_v == "SWIR":
        #     spec_image[:,:,78:86] = 0
        #     spec_image[:,:,154:270] = 

        meta = spec.metadata

        if s_v == "SWIR":
            first_spec = (spec_image[:,:,10]) 
        else:
            first_spec = (spec_image[:,:,249]) 

        
        first_spec = first_spec/first_spec.max()
        
    cv2.imwrite(result_dir+s_v +".png" , np.int64(first_spec*255))
    
    output_name_hdr = result_dir + "//__" + file +".hdr"
    
    return first_spec, spec_image, name_hdr, output_name_hdr



def read_igm(path_igm):
    '''
    Read igm coordinates file
    '''

    ls_dir = os.listdir(path_igm)
    for i,file in enumerate(ls_dir):
        if not file.endswith(".hdr"):
            continue

        name_hdr = path_igm + "//" + file
        spec = spectral.envi.open(name_hdr)
        spec_image = np.array(spec.asarray())
        lat_long = spec_image[:,:,0:2]
        lat_long  = lat_long.reshape((lat_long.shape[0] * lat_long.shape[1], lat_long.shape[2]))
        points= [Point([p[0],p[1]]) for p in lat_long ] 

        df = { 'geometry':points}
        gf = geopandas.GeoDataFrame(df)
        gf.set_crs(4326 , inplace = True)
        gf = gf.to_crs(32636)

        t = gf["geometry"].to_numpy()

        t = np.array([ np.array([p.x, p.y ]) for p in t ])
        utm_coords = t.reshape( ( spec_image.shape[0] , spec_image.shape[1] , 2)  )
        print(utm_coords.shape)


    return utm_coords, t


def find_closest_pixel(kdtree, pixel, swir_img, temp_vnir, vnir_img, coords_swir):
    '''
    Find the closest pixel to input pixel (SWIR and VNIR) by igm coords)
    '''
    
    points = np.array([coords_swir[pixel[0],pixel[1]]])
    closest_point = kdtree.query(points, k=1)[1][0]
    coords_closest = temp_vnir[closest_point]
    diff = [coords_closest[0] - points[0][0], coords_closest[1] - points[0][1]]
    
    rows = vnir_img.shape[0]
    cols = vnir_img.shape[1]
    col = int(closest_point % cols)
    row = int((closest_point - col)/cols)

    return [row, col], coords_closest, diff



def image_to_rgb (img):

    rgb  = np.zeros((img.shape[0] , img.shape[1] , 3))
    rgb[:,:,0] = img
    rgb[:,:,1] = img
    rgb[:,:,2] = img

    return rgb


def draw_points(pixel, rgb_swir, rgb_vnir, closest_pixel, flag_color=False):
    
    color = [random.random(), random.random(), random.random()]
    if flag_color:
        color = [255*random.random(), 255*random.random(), 255*random.random()]

    rgb_swir[pixel[0]-5:pixel[0]+5,pixel[1]-5:pixel[1]+5,:] = np.array(color)
    rgb_vnir[closest_pixel[0]-5:closest_pixel[0]+5,closest_pixel[1]-5:closest_pixel[1]+5,:] = np.array(color)

    return rgb_swir, rgb_vnir



def sort_fused_image_bands_and_make_hdr(full_image,name_hdr_swir,name_hdr_vnir,swir_img,vnir_img, flag_save=True):

    spec_swir = spectral.envi.open(name_hdr_swir)
    spec_vnir = spectral.envi.open(name_hdr_vnir)
    full_image_copy = full_image.copy()

    meta = spec_swir.metadata
    meta['samples'] = str(full_image.shape[1])
    meta['lines'] = str(full_image.shape[0])
    meta['bands'] = full_image.shape[2]
    meta2 = spec_vnir.metadata
    vnir_950 = vnir_img[:,:,249]
    swir_950 = swir_img[:,:,10]
    vnir_avg = np.median(vnir_img[:,:,226:270],axis=2)
    swir_avg = np.median(swir_img[:,:,2:17],axis=2)

    #vnir_950 = full_image[:,:,249]
    #swir_950 = full_image[:,:,283]

    meta_900_top =  meta['wavelength'][2:17]
    meta2_900_top =  meta2['wavelength'][226:270]
    meta['wavelength'] = np.concatenate((meta2['wavelength'],meta['wavelength']))
    wavelengths = meta['wavelength']
    wavelengths_sorted = np.sort(wavelengths)
    wave_indices = np.argsort(wavelengths, axis = 0)
  
    for i, index in enumerate(wave_indices):
        if i==index:
            continue
        else:
            full_image_copy[:,:,i] =  full_image[:,:,index]

    temp = []
    for i in range (len(wavelengths_sorted)):
        temp.append(str(wavelengths_sorted[i]))

    meta['wavelength'] = temp
    if flag_save:
        spectral.envi.save_image(output_name_hdr, full_image_copy, dtype=np.float32, metadata=meta, interleave="bsq",
                                 ext="", force=True)

    shared_waves = full_image_copy[:,:,228:288]
    mean_image = np.mean(shared_waves, axis = 2)
    


    return mean_image, vnir_avg, swir_avg, shared_waves



def fuse_by_igm(full_image, swir_img,vnir_img,kdtree, temp_vnir, coords_swir, rgb_swir, rgb_vnir):

    for i in range(0,swir_img.shape[0],1):
        print(i,swir_img.shape[0])
        for j in range(0,swir_img.shape[1],1): 
            pixel = [i,j]
            closest_pixel, closest_coords, diff = find_closest_pixel(kdtree, pixel, swir_img, temp_vnir, vnir_img, coords_swir)
            diffs.append(diff) 
            
            if i%500 == 0 and j%500 == 0:
                rgb_swir, rgb_vnir = draw_points(pixel, rgb_swir, rgb_vnir, closest_pixel) 
                
            
            full_image[i,j,0:full_vnir.shape[2]] = full_vnir[closest_pixel[0],closest_pixel[1]]
    

    return full_image, rgb_swir, rgb_vnir



def similar(tr1, tr2, threshold):

    if np.abs(tr1[0] - tr2[0]) < threshold and np.abs(tr1[1] - tr2[1]) < threshold:
        return True

    return False

def ransac_tr(transformations, flag_rigid, p1s):

    print("ransac_tr")
    p1s_new = []
    if flag_rigid:
        maxc = 0
        maxi = 0
        for i in range(len(transformations)):
            counter = 0
            for j in range(len(transformations)):
                if i == j:
                    continue
                
                if similar(transformations[i], transformations[j], threshold = 10):
                    counter+=1
                    if(counter > maxc):
                        maxc = counter
                        maxi = i
        
        return transformations[maxi]
    
    else:
        transformations_new = []
        indices_new = []
        maxc = 0
        maxi = 0
        for i in range(len(transformations)):
            counter = 0
            for j in range(len(transformations)):
                if i == j:
                    continue
                
                if similar(transformations[i], transformations[j], threshold = 20):
                    counter+=1
                    if(counter > maxc):
                        maxc = counter
                        maxi = i
        
        for j in range(len(transformations)):
            if similar(transformations[j], transformations[maxi], threshold = 20):
                transformations_new.append(list(transformations[j]))
                p1s_new.append(p1s[j])
                indices_new.append(j)

        
        return transformations[maxi], transformations_new, indices_new, p1s_new



def calc_transformation_by_closest_tr(pixel, transformations,p1s, kdtree):
    
    closest_point = kdtree.query(np.array(pixel), k=1)[1]
    tr = transformations[closest_point]
    closest_pixel = [pixel[0]-int(np.round(tr[0])), pixel[1]-int(np.round(tr[1]))]


    return closest_pixel, tr




def draw_hists(transformations_new):

    plt.figure()
    plt.hist(np.asarray(transformations_new)[:,0])
    plt.show()
    plt.title('transformations x')
    plt.xlabel('transformations x')
    plt.ylabel('count')
    plt.show()
    plt.savefig('transformations_x' + '.png', dpi = 400)

    plt.figure()
    plt.hist(np.asarray(transformations_new)[:,1])
    plt.show()
    plt.title('transformations y')
    plt.xlabel('transformations y')
    plt.ylabel('count')

    plt.show()
    plt.savefig('transformations_y' + '.png', dpi = 400)

    return


def draw_transformations(transformations_x, transformations_y, result_dir):

    min_x = np.min(transformations_x)
    min_y = np.min(transformations_y)

    if min_x < 0:
        transformations_x -= min_x
 
    if min_y < 0:
        transformations_y -= min_y
    
    max_y = np.max(transformations_y)
    max_x = np.max(transformations_x)

    transformations_x /= max_x
    transformations_y /= max_y
    
    rgb_trx_image = image_to_rgb (transformations_x)
    rgb_try_image = image_to_rgb (transformations_y)

    cv2.imwrite(result_dir+"transformations_x" +".png" ,np.int64(rgb_trx_image*255))
    cv2.imwrite(result_dir+"transformations_y" +".png" ,np.int64(rgb_try_image*255))

    return




def calc_polynomial(ps, ts):


    xs = np.array(ps)[:,0]
    ys = np.array(ps)[:,1]
    zs = np.array(ts)

    coeffs = np.ones((3,3))

    # solve array
    a = np.zeros((6, len(xs)))

    index_mapping = []
    # for each coefficient produce array x^i, y^j
    true_index = 0
    count = 0
    for index, (i, j) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        index_mapping.append((i,j,index))

        if  i + j <= 2:
            arr = coeffs[i, j] * xs**i * ys**j
            a[index - count] = arr.ravel()

        else: count+=1
   
   
    a=a.T
    eps = 1 
   # soln = np.dot(np.linalg.pinv(np.dot(a.T, a)+eps*np.identity(len(a[0]))),np.dot(a.T, np.ravel(zs)))
    soln = np.dot(np.linalg.pinv(np.dot(a.T, a)),np.dot(a.T, np.ravel(ts)))
    soln = list(soln)

    while len(soln) <  9:
        soln.append(0)

    soln = np.array(soln)

    soln_mat = np.zeros((3,3))
    count = 0
    for indmap in index_mapping:
        if indmap[0] + indmap[1] > 2:
            count += 1
            continue

        soln_mat[indmap[0],indmap[1]] = soln[indmap[2] - count]



    return soln_mat





def lstsq(indices1, indices2, transformations_new, points_xy):

    transformations_x = np.array(transformations_new)[:,0]
    transformations_y = np.array(transformations_new)[:,1]
    transformations_x = transformations_x[np.array(indices1)]
    transformations_y = transformations_y[np.array(indices2)]
    ps1 = np.array(points_xy)[indices1]
    ps2 = np.array(points_xy)[indices2]

    coeffs1 = calc_polynomial(ps1, transformations_x) 
    coeffs2 = calc_polynomial(ps2, transformations_y)


    return coeffs1, coeffs2



def ransac_polynomial (transformations_new, p1s_new):
    print("ransac_polynomial")
    max1 = 0
    max2 = 0
    final_good1 = []
    final_good2 = []
    final_coeffs1 = []
    final_coeffs2 = []

    for i in range (1000):    
        ps = []
        ts = []
        for j in range (5):
            r = int(len(transformations_new)*random.random())
            t = transformations_new [r]
            p = p1s_new[r]
            ps.append(p)
            ts.append(t)

        coeffs1 = calc_polynomial(ps, np.array(ts)[:,0]) 
        coeffs2 = calc_polynomial(ps, np.array(ts)[:,1])

        counter1 = 0
        counter2 = 0
        xs = np.array(ps)[:,0]
        ys = np.array(ps)[:,1]
       # zs1 = np.array(ts)[:,0]
        #zs2 = np.array(ts)[:,1]

        fitted_surf1 = np.polynomial.polynomial.polyval2d(np.array(p1s_new)[:,0], np.array(p1s_new)[:,1], coeffs1)
        fitted_surf2 = np.polynomial.polynomial.polyval2d(np.array(p1s_new)[:,0], np.array(p1s_new)[:,1], coeffs2)        
        deltas1 = np.abs(fitted_surf1 - np.array(transformations_new)[:,0])
        deltas2 = np.abs(fitted_surf2 - np.array(transformations_new)[:,1])
        good_indices_1 = np.where((deltas1<5))[0]
        good_indices_2 = np.where((deltas2<5))[0]
        count_good1 = len(good_indices_1)
        count_good2 = len(good_indices_2)
        if count_good1 > max1:
            max1 = count_good1
            final_good1 = good_indices_1
            final_coeffs1 = coeffs1

        if count_good2 > max2:
            max2 = count_good2
            final_good2 = good_indices_2
            final_coeffs2 = coeffs2




    return final_coeffs1, final_coeffs2, final_good1, final_good2


def calc_transformation_by_pixel(pixel ,coeffs1, coeffs2):
    fitted_surf1 = np.polynomial.polynomial.polyval2d(pixel[0], pixel[1], coeffs1)
    fitted_surf2 = np.polynomial.polynomial.polyval2d(pixel[0], pixel[1], coeffs2) 
    closest_pixel = [pixel[0]-int(np.round(fitted_surf1)), pixel[1]-int(np.round(fitted_surf2))]
    return [fitted_surf1, fitted_surf2], closest_pixel


def fuse_by_sift( swir_img, vnir_img, result_dir, full_swir, full_vnir, flag_rigid):
    
    print("Fuse by sift")
    sift = cv2.SIFT_create(nOctaveLayers = 120)
    #sift = cv2.SIFT_create()

    #sift = cv2.SIFT_create(sigma=10)
    # Define the neighborhood size

    # Iterate over every pixel in the image and compute the SIFT features
    keypoints = []
    descriptors = []
    transformations = []
   # Find keypoints and descriptors for both images
    swir_img = cv2.convertScaleAbs(swir_img*255)
    vnir_img = cv2.convertScaleAbs(vnir_img*255)
    rgb_swir = image_to_rgb (swir_img)
    vnir_img = cv2.flip(vnir_img,1)
    rgb_vnir = image_to_rgb (vnir_img)
    full_vnir = cv2.flip(full_vnir,1)
    full_image = np.zeros((full_swir.shape[0], full_swir.shape[1], full_swir.shape[2] + full_vnir.shape[2])) 
    full_image[:,:,full_vnir.shape[2]: full_swir.shape[2] + full_vnir.shape[2] ] = full_swir    
    # Compute the SIFT features for the neighborhood
    keypoints1, descriptors1 = sift.detectAndCompute(swir_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(vnir_img, None)

    if descriptors1 is None or descriptors2 is None :
        print(0)

    else:

        # Initialize the matcher
        matcher = cv2.FlannBasedMatcher()
        #matcher = cv2.BFMatcher()

        # Match descriptors
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        # Apply ratio test to filter good matches
        good_matches = []
        p1s = []
        for k in range(len(matches)):
            if len(matches[k]) < 2:
                continue
            else:
                m = matches[k][0]
                n = matches[k][1]
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

                    p1 = keypoints1[m.queryIdx].pt
                    p2 = keypoints2[m.trainIdx].pt
                    #rgb_swir[round(p1[1])-5:round(p1[1])+5 , round(p1[0])-5:round(p1[0])+5] = [255,0,0]
                    #rgb_vnir[round(p2[1])-5:round(p2[1])+5 , round(p2[0])-5:round(p2[0])+5] = [255,0,0]
                    #transformations.append(np.array([np.abs(p1[1]-p2[1]), np.abs(p1[0]-p2[0])]))
                    transformations.append(np.array([p1[1]-p2[1], p1[0]-p2[0]]))
                    p1s.append([p1[1],p1[0]])

        if flag_rigid:
            tr = ransac_tr(transformations, flag_rigid)
        else:
            tr, transformations_new, indices_new, p1s_new = ransac_tr(transformations, flag_rigid, p1s) 
            coeffs1, coeffs2, indices1, indices2 = ransac_polynomial (transformations_new, p1s_new)
            coeffs1, coeffs2 = lstsq(indices1, indices2, transformations_new, p1s_new) 
            good_matches = np.array(good_matches)[indices_new]
            p1s = np.array(p1s)[indices_new]
            #draw_hists(transformations_new)

        #cv2.imwrite(result_dir+"vnir_points_sift" +".png" ,np.int64(rgb_vnir))
        #cv2.imwrite(result_dir+"swir_points_sift" +".png" ,np.int64(rgb_swir))
        # Draw the matches
        #match_image = cv2.drawMatches(swir_img, keypoints1, vnir_img, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #print(len(good_matches))

        # Display the image with matches
        #cv2.imwrite(result_dir+"image matches" +".png" ,np.int64(match_image))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


    #transformations_x = np.zeros((swir_img.shape[0],swir_img.shape[1]))
    #transformations_y = np.zeros((swir_img.shape[0],swir_img.shape[1]))

    print("Till for " + str(time.time() - start) + " seconds")


    if flag_rigid:
        for i in tqdm.tqdm(range(0,swir_img.shape[0],1)):
            for j in range(0,swir_img.shape[1],1):     
                closest_pixel = [i-int(np.round(tr[0])), j-int(np.round(tr[1]))]
                closest_pixel[0] = max(closest_pixel[0], 0)
                closest_pixel[1] = max(closest_pixel[1], 0)
                closest_pixel[0] = min(closest_pixel[0], full_vnir.shape[0] - 1)
                closest_pixel[1] = min(closest_pixel[1], full_vnir.shape[1] - 1)
                full_image[i,j,0:full_vnir.shape[2]] = full_vnir[closest_pixel[0],closest_pixel[1]]

    else:

        for i in tqdm.tqdm(range(0,swir_img.shape[0],1)):
            for j in range(0,swir_img.shape[1],1):     
            
                tr_calculated, closest_pixel = calc_transformation_by_pixel([i,j],coeffs1, coeffs2)
                #transformations_x[i,j] = tr_calculated[0]
                #transformations_y[i,j] = tr_calculated[1]

                closest_pixel[0] = max(closest_pixel[0], 0)
                closest_pixel[1] = max(closest_pixel[1], 0)
                closest_pixel[0] = min(closest_pixel[0], full_vnir.shape[0] - 1)
                closest_pixel[1] = min(closest_pixel[1], full_vnir.shape[1] - 1)
                
                full_image[i,j,0:full_vnir.shape[2]] = full_vnir[closest_pixel[0],closest_pixel[1]]

    print("After for " + str(time.time() - start) + " seconds")

    #draw_transformations(transformations_x, transformations_y, result_dir)

    full_swir = full_image[:,:,full_vnir.shape[2]:full_image.shape[2]]
    full_vnir = full_image[:,:,0:full_vnir.shape[2]]

    return full_image, full_swir, full_vnir



def fix_image_by_vxvy(swir, vnir, vx, vy, full_image_old, full_swir, full_vnir, index):

    #full_image = full_image_old.copy()
    #full_image = full_image[:vnir.shape[0],:]

    vnir_new = vnir.copy()
    full_swir = full_swir[:vnir.shape[0],:]
    full_image = np.zeros((full_swir.shape[0], full_swir.shape[1], full_swir.shape[2] + full_vnir.shape[2])) 
    full_image[:,:,full_vnir.shape[2]: full_swir.shape[2] + full_vnir.shape[2] ] = full_swir    


    for i in range(0,full_image.shape[0],1):
        for j in range(0,full_image.shape[1],1):

            twin_pixel = [i+np.round(vx[i,j]), j+np.round(vy[i,j])]
            if twin_pixel[0] >= full_image.shape[0]:
                twin_pixel[0] = full_image.shape[0] - 1
            if twin_pixel[1] >= full_image.shape[1]:
                twin_pixel[1] = full_image.shape[1] -1
            
            full_image[i,j,0:full_vnir.shape[2]] = full_vnir[int(twin_pixel[0]),int(twin_pixel[1])]
            vnir_new[i,j] = vnir[int(twin_pixel[0]),int(twin_pixel[1])]

    full_swir = full_image[:,:,full_vnir.shape[2]:full_image.shape[2]]
    full_vnir = full_image[:,:,0:full_vnir.shape[2]]
    
    rgb_mean_image = image_to_rgb (vnir_new)
    cv2.imwrite(result_dir+"vnirnew" + str(index) + ".png" ,np.int64(rgb_mean_image*255))
    #rgb_mean_image = image_to_rgb (swir)
    #cv2.imwrite(result_dir+"swirnew" + str(index) +".png" ,np.int64(rgb_mean_image*255))

    return full_image , full_vnir, full_swir



def calc_loss (full_image, i):

    vnir_950 = full_image[:,:,249]
    swir_950 = full_image[:,:,283]


    delta = np.median(np.abs(vnir_950 - swir_950))

    return delta


def calc_m_f(full_image_shared):

    ms = []
    fs = []
    for i in range(0, full_image_shared.shape[2] - 1, 2):
        m = sp.ndimage.filters.gaussian_filter(full_image_shared[i],sigma=3)
        f = sp.ndimage.filters.gaussian_filter(full_image_shared[i+1],sigma=3)
        ms.append(m)
        fs.append(f)

    return ms, fs




def super_algorithm_new(swir, vnir, full_image, full_image_shared, full_swir, full_vnir, name_hdr_swir, name_hdr_vnir):

#    full_vnir = cv2.flip(full_vnir,1)
    #vnir = cv2.flip(vnir,1)

    if swir.shape[0] > vnir.shape[0] :
        swir = swir[:vnir.shape[0],:]

    swir = swir.reshape((swir.shape[0] , swir.shape[1]))
    vnir = vnir.reshape((vnir.shape[0] , vnir.shape[1]))
    
    points1 = []
    points2 = []
    for i in range(swir.shape[0]):
        for j in range(swir.shape[1]):
            points1.append([i,j,0])

    points1 = np.array(points1)
    points2 = points1.copy()

    orig_points1 = points1.copy()
    orig_points2 = points2.copy()
    m = sp.ndimage.filters.gaussian_filter(swir,sigma=3)
    f = sp.ndimage.filters.gaussian_filter(vnir,sigma=3)

    gf = np.gradient(f)
    gm = np.gradient(m)

    f = np.sqrt(gf[0]*gf[0] + gf[1]*gf[1])
    m = np.sqrt(gm[0]*gm[0] + gm[1]*gm[1])

    running_m = m.copy()
    running_f = f.copy()

    vx = np.zeros(swir.shape)
    vy = np.zeros(vnir.shape)
    full_image , full_vnir, full_swir = fix_image_by_vxvy(swir, vnir, vx, vy, full_image, full_swir, full_vnir, 0)
    y = np.linspace(0,swir.shape[0]-1,swir.shape[0])
    x = np.linspace(0,swir.shape[1]-1,swir.shape[1])
    xx,yy = np.meshgrid(x,y)

    field_m = sp.interpolate.RectBivariateSpline(x,y,m.T)
    field_m_swir = sp.interpolate.RectBivariateSpline(x,y,swir.T)
    deltas = []
    indices = []
    vx_no_norm = vx
    vy_no_norm = vy

    for i in range(300):
        print(i)
        # each iter max delta = 5cm
        
        #running_m_swir = field_m_swir.ev(xx+vx, yy+vy)
        #vnir

        m_m_f = running_m-running_f

        alpha = 0.8
        gf = np.gradient(running_f)

        mag_gf = gf[0]*gf[0] + gf[1]*gf[1]

        temp = m_m_f/(0.0001+mag_gf + alpha*alpha*(m_m_f*m_m_f))
        dvx = gf[0]*temp
        dvy = gf[1]*temp
        dvx = np.clip(dvx, -3, 3)
        dvy = np.clip(dvy, -3, 3)

        # TODO: WITHOUT NORMALIZE, BIGGER ALPHA, LESS ITERATIONS 
       
        vx = vx - dvy
        vy = vy - dvx

        vx = sp.ndimage.filters.gaussian_filter(vx,sigma=3)
        vy = sp.ndimage.filters.gaussian_filter(vy,sigma=3)


        max_x = xx.ravel().max()
        bad_x = (((xx+vx) > max_x) | ((xx+vx) < 0))
        vx[bad_x] = 0


        max_y = yy.ravel().max()
        bad_y = (((yy+vy) > max_y) | ((yy+vy) < 0))
        vy[bad_y] = 0
        running_m = field_m.ev(xx+vx, yy+vy)
        running_m = sp.ndimage.filters.gaussian_filter(running_m,sigma=3)
       
        #full_image , full_vnir, full_swir = fix_image_by_vxvy(swir, vnir, vx, vy, full_image, full_swir, full_vnir)
        #delta = calc_loss (full_image, i)  
        delt_mat = np.abs(running_m - running_f)
        delt_mat_new = delt_mat / np.percentile(delt_mat, 80)
        dvx = np.clip(dvx, 0, 1)

        delt_mat_new*=256
        
        delta = np.sum(delt_mat)
        # deltas.append(delta)
        # rgb_mean_image = image_to_rgb (delt_mat_new)
        # cv2.imwrite(result_dir+"deltas mat " + str(i) +".png" ,np.int64(rgb_mean_image*255))

        if (i+1)%100 ==0:
            plt.figure()
            plt.hist(np.sqrt(dvx*dvx+dvy*dvy))
            plt.show()
            plt.savefig('sqrt(dvx2+dvy2)' +str(i) +'.png', dpi = 400)  
            full_image, full_vnir, full_swir = fix_image_by_vxvy(swir, vnir, vx, vy, full_image, full_swir, full_vnir, i)

        #if (i+1)%100 == 0:
         #   full_image, full_vnir, full_swir = fix_image_by_vxvy(swir, vnir, vx, vy, full_image, full_swir, full_vnir, i)

    plt.figure()
    indices = range(len(deltas))
    plt.scatter(indices, deltas)
    plt.show()
    plt.savefig('deltas' + '.png', dpi = 400)

    #full_image , full_vnir, full_swir = fix_image_by_vxvy(swir, vnir, vx, vy, full_image, full_swir, full_vnir, i)
    #calc_loss (full_image, i)      
    #mean_image, vnir, swir  = sort_fused_image_bands_and_make_hdr(full_image,name_hdr_swir,name_hdr_vnir, full_swir, full_vnir)
    #rgb_mean_image = image_to_rgb (mean_image)
    #cv2.imwrite(result_dir+"combined demon " + str(i) +".png" ,np.int64(rgb_mean_image*255))


  
    return vx, vy, full_image 



def command_iteration(filter):
    print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")


def demons(swir, vnir, full_image, full_swir, full_vnir):

    print("demons")
    #fixed = sitk.ReadImage("swir_after_polynom.png", sitk.sitkFloat32)
    #moving = sitk.ReadImage("vnir_after_polynom.png", sitk.sitkFloat32)
    fixed = sitk.GetImageFromArray(swir)
    moving = sitk.GetImageFromArray(vnir)

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(200)
    matcher.SetNumberOfMatchPoints(50)
    moving = matcher.Execute(moving, fixed)

    # The basic Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in
    # SimpleITK
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(1000)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(1.0)
    demons.SetSmoothDisplacementField(True)
    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

    displacementField = demons.Execute(fixed, moving)
    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f" RMS: {demons.GetRMSChange()}")

    dis = displacementField
     
    vnir_new = vnir.copy()
    full_image = np.zeros((full_swir.shape[0], full_swir.shape[1], full_swir.shape[2] + full_vnir.shape[2])) 
    full_image[:,:,full_vnir.shape[2]: full_swir.shape[2] + full_vnir.shape[2] ] = full_swir    
    outTx = sitk.DisplacementFieldTransform(displacementField)

    full_image_new = full_image.copy()
    
    for i in tqdm.tqdm(range (full_vnir.shape[2])):
        moving =  full_vnir[:,:,i]
        moving = sitk.GetImageFromArray(moving)
        t = sitk.Resample(moving,fixed,outTx, sitk.sitkBSpline,0.0,moving.GetPixelID())
        tt = sitk.GetArrayViewFromImage(t)
        full_image_new[:,:,i] = tt
    
    rgb_mean_image = image_to_rgb (tt)
    cv2.imwrite(result_dir+"vnir_new" + str(i) +".png" ,np.int64(rgb_mean_image*255))


    # if "SITK_NOSHOW" not in os.environ:
    #     resampler = sitk.ResampleImageFilter()
    #     resampler.SetReferenceImage(fixed)
    #     resampler.SetInterpolator(sitk.sitkLinear)
    #     resampler.SetDefaultPixelValue(100)
    #     resampler.SetTransform(outTx)

    #     out = resampler.Execute(moving)
        
    #     simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    #     simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    #     # Use the // floor division operator so that the pixel type is
    #     # the same for all three images which is the expectation for
    #     # the compose filter.
    #     cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    #     n = sitk.GetArrayFromImage(cimg)
    #     cv2.imwrite(result_dir+"vnirtry2" +".png" ,np.int64(n))
    
   
    # full_image_new = full_image.copy()
  

    return full_image_new




if __name__ == '__main__':

    start = time.time()
    path_images = "input_data/"
    path_swir = os.path.join(path_images, "SWIR/2")
    path_vnir = os.path.join(path_images, "VNIR/2")
    flag_igm = False
    flag_make_pickles = False
    result_dir = "input_data" + "/results/"
    name_hdr = "input_data" + "//output"
    
    swir_img, full_swir, name_hdr_swir, output_name_hdr = read_images(path_swir,"SWIR",result_dir)
    vnir_img, full_vnir, name_hdr_vnir, output_name_hdr = read_images(path_vnir,"VNIR",result_dir)
    rgb_swir = image_to_rgb (swir_img)
    rgb_vnir = image_to_rgb (vnir_img)
    full_image = np.zeros((full_swir.shape[0], full_swir.shape[1], full_swir.shape[2] + full_vnir.shape[2])) 
    full_image[:,:,full_vnir.shape[2]: full_swir.shape[2] + full_vnir.shape[2] ] = full_swir
    diffs = []

    if flag_igm:
        path_igm_vnir = os.path.join(path_images, "vnir/igm")
        path_igm_swir = os.path.join(path_images, "swir/igm")
        if flag_make_pickles:
            coords_vnir, temp_vnir = read_igm(path_igm_vnir)
            coords_swir, temp_swir = read_igm(path_igm_swir)
            vnir = (coords_vnir, temp_vnir)
            swir = (coords_swir, temp_swir)

            with open('vnir.pkl', "wb") as f:
                pickle.dump(vnir, f)
            
            with open('swir.pkl', "wb") as f:
                pickle.dump(swir, f)
        else:
            with open('swir.pkl', "rb") as f:
                swir = pickle.load(f)
            with open('vnir.pkl', "rb") as f:
                vnir = pickle.load(f)

        coords_vnir = vnir[0]
        temp_vnir = vnir[1]
        coords_swir = swir[0]
        temp_swir = swir[1]
        kdtree = KDTree(temp_vnir)
        full_image, rgb_swir, rgb_vnir = fuse_by_igm(full_image, swir_img,vnir_img,kdtree, temp_vnir, coords_swir, rgb_swir, rgb_vnir)
        cv2.imwrite(result_dir+"vnir_points" +".png" ,np.int64(rgb_vnir*255))
        cv2.imwrite(result_dir+"swir_points" +".png" ,np.int64(rgb_swir*255))
        mean_image, vnir, swir = sort_fused_image_bands_and_make_hdr(full_image,name_hdr_swir,name_hdr_vnir)
        rgb_mean_image = image_to_rgb (mean_image)
        cv2.imwrite(result_dir+"combined" +".png" ,np.int64(rgb_mean_image*255))
        
    else:
        full_image, full_swir, full_vnir = fuse_by_sift(swir_img,vnir_img, result_dir, full_swir, full_vnir, False)
        mean_image, vnir, swir, full_image_shared  = sort_fused_image_bands_and_make_hdr(full_image,name_hdr_swir,name_hdr_vnir, full_swir, full_vnir, flag_save=False)
        #rgb_mean_image = image_to_rgb (mean_image)
        #cv2.imwrite(result_dir+"combined_after_polynom" +".png" ,np.int64(rgb_mean_image*255))
        #rgb_mean_image = image_to_rgb (vnir)
        #cv2.imwrite(result_dir+"vnir_after_polynom" +".png" ,np.int64(rgb_mean_image*255))
        #rgb_mean_image = image_to_rgb (swir)
        #cv2.imwrite(result_dir+"swir_after_polynom" +".png" ,np.int64(rgb_mean_image*255))
        print("Till demons " + str(time.time() - start) + " seconds")

        full_image_new = demons(swir, vnir, full_image, full_swir, full_vnir)
        mean_image, vnir, swir, full_image_shared  = sort_fused_image_bands_and_make_hdr(full_image_new,name_hdr_swir,name_hdr_vnir, full_swir, full_vnir)
        rgb_mean_image = image_to_rgb (mean_image)
        cv2.imwrite(result_dir+"combined_demon" +".png" ,np.int64(rgb_mean_image*255))
    
    
    print("Total " + str(time.time() - start) + " seconds")
    print('END')


