import os
import cv2
import numpy as np
import pickle as pkl


def crop_image(path, file_name):
    file = open(path + 'labels/' + file_name, 'r')
    line = file.readlines()
    file.close()
    numbers = line[0].split()
    numbers = [float(x) for x in numbers]

    image = cv2.imread(path + 'images/' + file_name[:-3] + 'jpg')
    x = numbers[1] * image.shape[1]
    y = numbers[2] * image.shape[0]
    width = numbers[3] * image.shape[1]
    height = numbers[4] * image.shape[0]

    cropped_image = image[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
    name = file_name.split("_")
    resized_image = cv2.resize(cropped_image, (640, 640))
    cv2.imwrite('crop/' + name[0] + ".jpg", resized_image)
    return resized_image


def RGN2NVDI(image):
    red_band = image[:, :, 0].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_normalized = (ndvi + 1) * 127.5
    return ndvi_normalized.astype(np.uint8)


def normal(image, a=1 / 100, b=15):
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

            image_matrix[i][j] = image_matrix[i][j] + (image_matrix[i][j] - mean) * (a * mean) + b

            if image_matrix[i][j] > 255:
                image_matrix[i][j] = 255

    return image_matrix


def RGN2NVDI_green(image):
    red_band = image[:, :, 1].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_normalized = (ndvi + 1) * 127.5
    return ndvi_normalized.astype(np.uint8)


def RGN2NVDI_green_segment(image, file_name):
    red_band = image[:, :, 1].astype(float)
    nir_band = image[:, :, 2].astype(float)
    if (file_name[0] == 'r'):
        b = 0
        a = 1 / 100
    elif (file_name[0] == 'y'):
        b = 5
        a = 1 / 300
    else:
        a = 1 / 100
        b = 15
    ndvi = (nir_band - red_band - 0.0001) / (nir_band + red_band + 0.0001)
    ndvi_normalized = (ndvi + 1) * 127.5
    ndvi_normalized = ndvi_normalized.astype(np.uint8)
    ndvi_normalized = normal(ndvi_normalized, a, b)
    file_name = file_name + 'jpg'
    cv2.imwrite('GNDVI/' + file_name, cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET))
    return ndvi_normalized


def RGN2NVDI_segment(image, file_name):
    if (file_name[0] == 'r'):
        b = 15
        a = 1 / 150
    elif (file_name[0] == 'y'):
        b = 40
        a = 1 / 500
    else:
        a = 1 / 200
        b = 40
    red_band = image[:, :, 0].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band - 0.0001) / (nir_band + red_band + 0.0001)
    ndvi_normalized = (ndvi + 1) * 127.5
    ndvi_normalized = ndvi_normalized.astype(np.uint8)
    ndvi_normalized = normal(ndvi_normalized, a, b)
    file_name = file_name + '.jpg'
    # cv2.imwrite('D:/FinalProject/NDVI/' + file_name,cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET))
    cv2.imwrite('E:/final_project_main/NDVI/' + file_name, ndvi_normalized)
    return ndvi_normalized


def segment(image, file_name):
    NVDI = RGN2NVDI(image)
    NVDI_gr = RGN2NVDI_green(image)

    _, segmented_image = cv2.threshold(NVDI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image = np.logical_not(segmented_image).astype(int)
    segmented_image = segmented_image.astype(np.uint8)

    _, segmented_image_gr = cv2.threshold(NVDI_gr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image_gr = np.logical_not(segmented_image_gr).astype(int)
    segmented_image_gr = segmented_image_gr.astype(np.uint8)

    mask = cv2.bitwise_and(segmented_image_gr, segmented_image_gr, mask=segmented_image)
    result = cv2.bitwise_and(image, image, mask=mask)
    # print(file_name)
    file_name = file_name + '.jpg'
    cv2.imwrite('/segment/' + file_name, result)
    return result

X = []
y = []
if __name__ == '__main__':
    path = "E:/Code/final_project-main/raw_data/"
    file_names = os.listdir(path + 'labels/')
    for file_name in file_names:
        image = crop_image(path, file_name)
        name = file_name.split("_")
        image_segment = segment(image, name[0])
        NDVI_segment = RGN2NVDI_segment(image_segment, name[0])
        hist, bins = np.histogram(NDVI_segment.flatten(), bins=256, range=[20, 240])
        norm_hist = hist / sum(hist)
        X.append(norm_hist)
        if(file_name[0] == 'g'): y.append(0)
        elif(file_name[0] == 'r'): y.append(1)
        else: y.append(2)
    
    with open("train_vege1.pkl", "wb") as f:
        pkl.dump([X, y], f)