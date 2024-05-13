import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def normal(image, a=1/100, b = 15):
    # Chuyển đổi hình ảnh thành ma trận
    image_matrix = np.array(image)
    # hist, bins = np.histogram(image.flatten(), bins=256, range=[20, 120])
    mean = int(np.mean(image_matrix))

    # Áp dụng hàm bậc 2 vào từng phần tử trong ma trận
    rows, cols = image_matrix.shape
    for i in range(rows):
        for j in range(cols):
            if image_matrix[i][j] < 5:
                continue
            if image_matrix[i][j] > 230:
                continue
            
            image_matrix[i][j] = image_matrix[i][j] + (image_matrix[i][j] - mean) * ( a * mean ) + b
            
            if image_matrix[i][j] > 255:
                image_matrix[i][j] = 255

    return image_matrix

def RGN2NVDI_green(image, cl = 'r'):
    red_band = image[:, :, 1].astype(float)
    nir_band = image[:, :, 2].astype(float)
    if(cl == 'r'): 
        b = 0
        a = 1/100
    elif (cl == 'y'):
        b = 5
        a = 1/300
    else:
        a = 1/100
        b = 15
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_normalized = (ndvi + 1) * 127.5
    ndvi_normalized = ndvi_normalized.astype(np.uint8)
    ndvi_normalized = normal(ndvi_normalized,a,b)
    return ndvi_normalized

def RGN2NVDI(image, cl = 'r'):
    if(cl == 'r'): 
        b = 0
        a = 1/100
    elif (cl == 'y'):
        b = 15
        a = 1/300
    else:
        a = 1/100
        b = 15
    red_band = image[:, :, 0].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_normalized = (ndvi + 1) * 127.5
    ndvi_normalized = ndvi_normalized.astype(np.uint8)
    ndvi_normalized = normal(ndvi_normalized,a,b)
    return ndvi_normalized


def histogram(image, file_name):
    # Tính histogram của ảnh
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Hiển thị histogram
    plt.figure(figsize=(10, 5))
    plt.hist(image.flatten(), bins=256, range=[20, 230], color='gray')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of the Image')
    

# image = cv2.imread("D:/FinalProject/segment/green90.JPG")
# NVDI = RGN2NVDI(image, 'r')
# NVDI_gr = RGN2NVDI(image, 'y')
# color_map2 = cv2.applyColorMap(NVDI, cv2.COLORMAP_JET)
# color_map1 = cv2.applyColorMap(NVDI_gr, cv2.COLORMAP_JET)

# cv2.imshow("img",NVDI)
# cv2.imshow("color_map1",color_map1)
# cv2.imshow("color_map2",color_map2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


folder_path = "E:/Code/final_project-main/crop/"
folder_write_path = "E:/Code/final_project-main/NDVI/"
file_names = os.listdir(folder_path)
for file_name in file_names:
    if(file_name[0] == 'r'):
        a = 1/100
        b = 15
    image = cv2.imread(folder_path + file_name)  
    image[:,:,2], image[:,:,0] = image[:,:,0], RGN2NVDI(image,file_name[0])
    # NVDI_gr = RGN2NVDI_green(image)

    # result = normal(NVDI,file_name[0])
    cv2.imwrite(folder_write_path + file_name,image)