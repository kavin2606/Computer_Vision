import cv2
import numpy as np
import sys

def convert_color_space_BGR_to_RGB(img_BGR):
    
    #print(img_BGR)
    B = img_BGR[:,:,0]/255
    G = img_BGR[:,:,1]/255
    R = img_BGR[:,:,2]/255
    img_RGB = np.stack([R, G, B], axis=2)
    
    #print(img_RGB)
    
    return img_RGB

def convert_color_space_RGB_to_BGR(img_RGB):
    
    R = source[:,:,0]*255
    G = source[:,:,1]*255
    B = source[:,:,2]*255
    img_BGR = np.stack([B, G, R], axis=2)
    
    return img_BGR

def convert_color_space_RGB_to_Lab(img_RGB):
    '''
        convert image color space RGB to Lab
        '''
    
    #print(img_RGB)
    R = img_RGB[:,:,0]
    G = img_RGB[:,:,1]
    B = img_RGB[:,:,2]
    
    L = 0.3811*R + 0.5783*G + 0.0402*B
    M = 0.1967*R + 0.7244*G + 0.0782*B
    S = 0.0241*R + 0.1288*G + 0.8444*B
    
    L = np.log10(L)
    M = np.log10(M)
    S = np.log10(S)
    
    new_l = 1.0 / np.sqrt(3)*L + 1.0 / np.sqrt(3)*M + 1.0 / np.sqrt(3)*S
    new_alpha = 1.0 / np.sqrt(6)*L + 1.0 / np.sqrt(6)*M - 2 / np.sqrt(6)*S
    new_beta = 1.0 / np.sqrt(2)*L - 1.0 / np.sqrt(2)*M + 0 *S
    
    img_Lab = np.stack([new_l, new_alpha, new_beta], axis=2)
    #print(img_Lab)

    return img_Lab

def convert_color_space_Lab_to_BGR(img_Lab):
    '''
        convert image color space Lab to RGB
        '''
    l_result = img_Lab[:,:,0]
    alpha_result = img_Lab[:,:,1]
    beta_result = img_Lab[:,:,2]
    
    L = np.sqrt(3.0) / 3.0 * l_result + np.sqrt(6) / 6.0 * alpha_result + np.sqrt(2) / 2.0 * beta_result
    M = np.sqrt(3.0) / 3.0 * l_result + np.sqrt(6) / 6.0 * alpha_result - np.sqrt(2) / 2.0 * beta_result
    S = np.sqrt(3.0) / 3.0 * l_result - np.sqrt(6) / 3.0 * alpha_result - 0 * beta_result


    L = np.power(10.0, L)
    M = np.power(10.0, M)
    S = np.power(10.0, S)

    R = 4.4679*L - 3.5873*M + 0.1193*S
    G = -1.2186*L + 2.3809*M - 0.1624*S
    B = 0.0497*L - 0.2439*M + 1.2045*S

    R = R*255
    G = G*255
    B = B*255

    img_BGR = np.stack([B, G, R], axis=2)

#print(img_BGR)
    return img_BGR

def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
        convert image color space RGB to CIECAM97s
        '''
    R = img_RGB[:,:,0]
    G = img_RGB[:,:,1]
    B = img_RGB[:,:,2]
        
    L = 0.3811*R + 0.5783*G + 0.0402*B
    M = 0.1967*R + 0.7244*G + 0.0782*B
    S = 0.0241*R + 0.1288*G + 0.8444*B
    
    A = 2.00*L + 1.00*M + 0.05*S
    C1 = 1.00*L - 1.09*M + 0.09*S
    C2 = 0.11*L + 0.11*M - 0.22*S
    
    img_CIECAM97s = np.stack([A, C1, C2], axis=2)
        
    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
        convert image color space CIECAM97s to RGB
        '''
    A_result = img_CIECAM97s[:,:,0]
    C1_result = img_CIECAM97s[:,:,1]
    C2_result = img_CIECAM97s[:,:,2]
    
    L = 0.32786885*A_result + 0.32159385*C1_result + 0.20607677*C2_result
    M = 0.32786885*A_result - 0.63534395*C1_result - 0.18539779*C2_result
    S = 0.32786885*A_result - 0.15687505*C1_result - 4.53511505*C2_result

    R = 4.4679*L - 3.5873*M + 0.1193*S
    G = -1.2186*L + 2.3809*M - 0.1624*S
    B = 0.0497*L - 0.2439*M + 1.2045*S
    
    R = R * 255
    G = G * 255
    B = B * 255
    
    img_RGB = np.stack([B, G, R], axis=2)
    
    return img_RGB


def color_transfer_in_Lab(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    
    #print(img_RGB_source)
    #print(img_RGB_target)
    source_lab = convert_color_space_RGB_to_Lab(img_RGB_source)
    target_lab = convert_color_space_RGB_to_Lab(img_RGB_target)
    #print(source_lab)
    #print(target_lab)
    l_source = source_lab[:,:,0]
    a_source = source_lab[:,:,1]
    b_source = source_lab[:,:,2]

    l_target = target_lab[:,:,0]
    a_target = target_lab[:,:,1]
    b_target = target_lab[:,:,2]

    l_source_mean = l_source - np.mean(l_source)
    alpha_source_mean = a_source - np.mean(a_source)
    beta_source_mean = b_source - np.mean(b_source)

    l_source_std = np.std(l_source)
    alpha_source_std = np.std(a_source)
    beta_source_std = np.std(b_source)

    l_target_mean = np.mean(l_target)
    alpha_target_mean = np.mean(a_target)
    beta_target_mean = np.mean(b_target)

    l_target_std = np.std(l_target)
    alpha_target_std = np.std(a_target)
    beta_target_std = np.std(b_target)

    l_result = (l_target_std/l_source_std) * l_source_mean
    alpha_result = (alpha_target_std/alpha_source_std) * alpha_source_mean
    beta_result = (beta_target_std/beta_source_std) * beta_source_mean
    

    l_result += l_target_mean
    alpha_result += alpha_target_mean
    beta_result += beta_target_mean

    img_trans = np.stack([l_result, alpha_result, beta_result], axis=2)
    #print(img_trans)
    img_conv_BGR = convert_color_space_Lab_to_BGR(img_trans)
    #print("inside",img_conv_BGR)
    return img_conv_BGR



def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    
    R = img_RGB_source[:,:,0]
    G = img_RGB_source[:,:,1]
    B = img_RGB_source[:,:,2]
    
    r_source_mean = R - np.mean(R)
    g_source_mean = G - np.mean(G)
    b_source_mean = B - np.mean(B)

    r_source_std = np.std(R)
    g_source_std = np.std(G)
    b_source_std = np.std(B)

    R_tar = img_RGB_target[:,:,0]
    G_tar = img_RGB_target[:,:,1]
    B_tar = img_RGB_target[:,:,2]

    r_target_mean = np.mean(R_tar)
    g_target_mean = np.mean(G_tar)
    b_target_mean = np.mean(B_tar)

    r_target_std = np.std(R_tar)
    g_target_std = np.std(G_tar)
    b_target_std = np.std(B_tar)

    r_result = (r_target_std/r_source_std) * r_source_mean
    g_result = (g_target_std/g_source_std) * g_source_mean
    b_result = (b_target_std/b_source_std) * b_source_mean

    r_result += r_target_mean
    g_result += g_target_mean
    b_result += b_target_mean
    r_result = r_result*255
    g_result = g_result*255
    b_result = b_result*255

    result_img = np.stack([b_result, g_result, r_result], axis=2)
    return result_img

def color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_CIECAM97s =====')
    source = convert_color_space_RGB_to_CIECAM97s(img_RGB_source)
    target = convert_color_space_RGB_to_CIECAM97s(img_RGB_target)
    A_source = source[:,:,0]
    C1_source = source[:,:,1]
    C2_source = source[:,:,2]

    A_target = target[:,:,0]
    C1_target = target[:,:,1]
    C2_target = target[:,:,2]

    A_source_mean = A_source - np.mean(A_source)
    C1_source_mean = C1_source - np.mean(C1_source)
    C2_source_mean = C2_source - np.mean(C2_source)
    
    A_source_std = np.std(A_source)
    C1_source_std = np.std(C1_source)
    C2_source_std = np.std(C2_source)

    A_target_mean = np.mean(A_target)
    C1_target_mean = np.mean(C1_target)
    C2_target_mean = np.mean(C2_target)
    
    A_target_std = np.std(A_target)
    C1_target_std = np.std(C1_target)
    C2_target_std = np.std(C2_target)

    A_result = (A_target_std/A_source_std) * A_source_mean
    C1_result = (C1_target_std/C1_source_std) * C1_source_mean
    C2_result = (C2_target_std/C2_source_std) * C2_source_mean

    A_result += A_target_mean
    C1_result += C1_target_mean
    C2_result += C2_target_mean

    img_trans = np.stack([A_result, C1_result, C2_result], axis=2)
    #print(img_trans)
    img_conv_BGR = convert_color_space_CIECAM97s_to_RGB(img_trans)
    #print("inside",img_conv_BGR)
    return img_conv_BGR


def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
#print(img_RGB_new)
    return img_RGB_new

if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2019, HW1: color transfer')
    print('==================================================')
    
    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]
    
    source = cv2.imread(path_file_image_source)
    target = cv2.imread(path_file_image_target)
    
    img_RGB_source = convert_color_space_BGR_to_RGB(source)
    img_RGB_target = convert_color_space_BGR_to_RGB(target)
    
    
    # ===== read input images
    # img_RGB_source: is the image you want to change the its color
    # img_RGB_target: is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)
    
    img_RGB_new_Lab       = color_transfer(img_RGB_source, img_RGB_target, option='in_Lab')
    #print("final",img_RGB_new_Lab)
    
    
    
    cv2.imwrite(path_file_image_result_in_Lab, img_RGB_new_Lab)
    
    img_RGB_new_RGB       = color_transfer(img_RGB_source, img_RGB_target, option='in_RGB')
    
    cv2.imwrite(path_file_image_result_in_RGB, img_RGB_new_RGB)
    # todo: save image to path_file_image_result_in_RGB
    
    img_RGB_new_CIECAM97s = color_transfer(img_RGB_source, img_RGB_target, option='in_CIECAM97s')
    cv2.imwrite(path_file_image_result_in_CIECAM97s, img_RGB_new_CIECAM97s)
# todo: save image to path_file_image_result_in_CIECAM97s

