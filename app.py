import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
from datetime import date
import base64
import os
import paho.mqtt.client as mqtt
import json
import RPi.GPIO as GPIO #Run tren Raspberry Pi
import time

THINGSBOARD_HOST = 'thingsboard.dke.vn'
ACCESS_TOKEN = 'pKEviMebI0ZHcU7QVWu8'
TOPIC = 'v1/devices/me/telemetry'

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))


def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))


current_folder_path = os.getcwd()
current_folder_path = current_folder_path.replace("\\", "/")
image = None
file_path = None
result = None
name = None

def send_data_to_thingsboard(data):
    client = mqtt.Client()
    client.username_pw_set(ACCESS_TOKEN)
    client.connect(THINGSBOARD_HOST, 1883, 60)
    client.publish(TOPIC, json.dumps(data))
    client.disconnect()

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

def crop_image(file_name):
    try:
        file = open(file_name[:-3]+'txt', 'r')
        line = file.readlines()
        file.close()
        numbers = line[0].split()
        numbers = [float(x) for x in numbers]

        image = cv2.imread(file_name)
        x = numbers[1] * image.shape[1]
        y = numbers[2] * image.shape[0]
        width = numbers[3] * image.shape[1]
        height = numbers[4] * image.shape[0]

        cropped_image = image[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
        name = file_name.split("/")
        resized_image = cv2.resize(cropped_image, (640,640))
        cv2.imwrite('crop/'+name[-1], resized_image)
        return Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    except FileNotFoundError:
        # Nếu không tìm thấy file txt tương ứng, mở file cố định
        default_file = open('E:/Code/final_project-main/raw_data/images/green_2019_0101_003937_814_JPG.rf.ddcfb897d4f4dd1e60ed164a5dc1adc8 (1_1).txt', 'r')
        default_line = default_file.readlines()
        default_file.close()
        default_numbers = default_line[0].split()
        default_numbers = [float(x) for x in default_numbers]

        image = cv2.imread(file_name)
        x = default_numbers[1] * image.shape[1]
        y = default_numbers[2] * image.shape[0]
        width = default_numbers[3] * image.shape[1]
        height = default_numbers[4] * image.shape[0]

        cropped_image = image[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
        name = file_name.split("/")
        resized_image = cv2.resize(cropped_image, (640,640))
        cv2.imwrite('crop/'+name[-1], resized_image)
        return Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

def segment(image, file_path):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    NVDI = RGN2NVDI(image)
    NVDI_gr = RGN2NVDI_green(image)

    _, segmented_image = cv2.threshold(NVDI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image = np.logical_not(segmented_image).astype(int)
    segmented_image = segmented_image.astype(np.uint8)

    _, segmented_image_gr = cv2.threshold(NVDI_gr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image_gr = np.logical_not(segmented_image_gr).astype(int)
    segmented_image_gr = segmented_image_gr.astype(np.uint8)

    mask = cv2.bitwise_and(segmented_image_gr,segmented_image_gr,mask=segmented_image)
    result = cv2.bitwise_and(image,image,mask=mask)
    # print(file_name)
    file_name = file_path.split("/")
    cv2.imwrite('/segment/' + file_name[-1], result)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

def RGN2NVDI_segment(image, file_path):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    file_name = file_path.split("/")
    if(file_name[-1][0] == 'r'): 
        b = 15
        a = 1/150
    elif (file_name[-1][0] == 'y'):
        b = 40
        a = 1/500
    else:
        a = 1/200
        b = 40
    red_band = image[:, :, 0].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band - 0.0001) / (nir_band + red_band + 0.0001)
    ndvi_normalized = (ndvi + 1) * 127.5
    ndvi_normalized = ndvi_normalized.astype(np.uint8)
    ndvi_normalized = normal(ndvi_normalized,a,b)
    # file_name = file_path.split("/")

    color_map = cv2.COLORMAP_JET  # Chọn bản đồ màu từ xanh đến đỏ
    color_mapped_image = cv2.applyColorMap(ndvi_normalized, color_map)  # Áp dụng bản đồ màu lên ảnh
    cv2.imwrite(current_folder_path + '/NDVI/' + file_name[-1],color_mapped_image)
    return ndvi_normalized


def encode_image_to_string(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def decode_image(image_string):
    decoded_image = base64.b64decode(image_string)
    nparr = np.frombuffer(decoded_image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# file_path = None
image = None
file_path = None
result = None
name = None
status = None

def tab2():
    # Function to handle Button 1 click
    def button1_click():
        # M? h?p tho?i ch?n file
        global image
        global file_path
        global current_date
        global name
        name = 'cay1'
        current_date = date.today()
        file_path = filedialog.askopenfilename()
        file_path = current_folder_path + '/raw_data' + file_path.split('raw_data')[1]

        label2 = ttk.Label(tab2_frame, text="Tr?ng thái")
        label2.pack(pady=10)
        label2.place(x=270, y=62, width=70, height=23)

        label6 = ttk.Label(tab2_frame, text='Ngày:  ')
        label6.pack(pady=10)
        label6.place(x=270, y=5, width=70, height=23)

        label6 = ttk.Label(tab2_frame, text=current_date)
        label6.pack(pady=10)
        label6.place(x=360, y=5, width=70, height=23)

        # Create a Label
        label4 = ttk.Label(tab2_frame, text="Cây")
        label4.pack(pady=10)
        label4.place(x=270, y=40, width=70, height=23)

        def on_combobox_select(event):
            global name
            name = 'cay' + combobox.get()
            print("Selected item:", name)

        combobox = ttk.Combobox(tab2_frame, values=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
        # ??t giá tr? m?c ??nh cho Combobox
        combobox.set("Ch?n tên cây")
        # ??t v? trí và kích th??c cho Combobox
        combobox.pack(pady=10)
        # G?n s? ki?n khi ch?n m?c trong Combobox
        combobox.place(x=370, y=40, width=70, height=23)
        combobox.bind("<<ComboboxSelected>>", on_combobox_select)
        # Ki?m tra xem ng??i dùng ?ã ch?n file hay ch?a

        if file_path:
            image = crop_image(file_path)
            image = image.resize((250, 250))
            photo = ImageTk.PhotoImage(image)
            # label.place(x=17, y=17, width=300, height=300)
            label = tk.Label(tab2_frame, image=photo)
            label.place(x=5, y=5)
            # label.pack()
            tab2_frame.mainloop()

    # Function to handle Button 2 click
    def button2_click():
        global image
        global file_path
        global result
        global status
        # Create a Label
        label3 = ttk.Label(tab2_frame, text="")
        label3.pack(pady=10)
        label3.place(x=370, y=62, width=70, height=23)
        if image:
            image_seg = segment(image, file_path)
            NDVI_segment = RGN2NVDI_segment(image_seg, file_path)
            hist, bins = np.histogram(NDVI_segment.flatten(), bins=256, range=[20, 240])
            plt.plot(hist)
            # Save the plot as an image
            plt.savefig(file_path[:-3] + 'png')
            # Open the saved image using PIL
            saved_image = Image.open(file_path[:-3] + 'png')
            saved_image = saved_image.resize((200, 170))
            norm_hist = hist / sum(hist)
            loaded_model = pkl.load(open("svm_model.pickle", "rb"))
            result = loaded_model.predict([norm_hist])
            if result == 0:
                status = "T?t"
            elif result == 1:
                status = "Thi?u n??c"
            else:
                status = "Bình th??ng"
            label3.config(text=status)

            # print(file_path.replace("raw_data/images", "NDVI"))
            NDVI_img = Image.open(file_path.replace("raw_data/images", "NDVI"))
            NDVI_img = NDVI_img.resize((200, 170))
            photo = ImageTk.PhotoImage(NDVI_img)
            label = tk.Label(tab2_frame, image=photo)
            label.place(x=265, y=85)
            tab2_frame.mainloop()

    # Function to handle Button 2 click
    def button3_click():
        global image
        global file_path
        global result
        global name

        current_date = str(date.today()).replace('-', '_')

        client = mqtt.Client()
        client.username_pw_set(ACCESS_TOKEN)
        client.on_connect = on_connect
        client.on_message = on_message

        client.connect(THINGSBOARD_HOST, 1883, 60)
        client.loop_start()

        data = {
            'name': name,
            'date': current_date,
            'position': "Cau Giay",
            'status': status
        }

        client.publish(TOPIC, json.dumps(data))
        print("Data sent", data)

        # send_data_to_thingsboard(data)
        client.loop_stop()
        client.disconnect()
        
        # send_data_to_thingsboard(data)
        label1 = ttk.Label(tab2_frame, text="L?u tr? thành công!")
        label1.pack(pady=10)
        label1.place(x=410, y=275, width=175, height=23)

    def button4_click():
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)
        GPIO.output(17, 1)
        time.sleep(0.002)
        GPIO.cleanup()

        time.sleep(0.1)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(17, GPIO.OUT)
        time.sleep(0.001)
        GPIO.cleanup()

    # Create Button 1
    button1 = ttk.Button(tab2_frame, text="M?", command=button1_click)
    button1.pack(pady=10)
    button1.place(x=17, y=275, width=75, height=23)

    # Create Button 2
    button2 = ttk.Button(tab2_frame, text="Phân tích", command=button2_click)
    button2.pack(pady=10)
    button2.place(x=110, y=275, width=75, height=23)

    # Create Button 3
    button3 = ttk.Button(tab2_frame, text="Thu d? li?u", command=button4_click)
    button3.pack(pady=10)
    button3.place(x=205, y=275, width=75, height=23)

    button4 = ttk.Button(tab2_frame, text="Luu tr?", command=button3_click)
    button4.pack(pady=10)
    button4.place(x=305, y=275, width=75, height=23)



# Create the main application window
root = tk.Tk()
root.title("Theo dõi cây trồng")
root.geometry("553x336")

# Create a Tab Control
tab_control = ttk.Notebook(root)

# Create the second tab
tab2_frame = ttk.Frame(tab_control)
tab_control.add(tab2_frame, text='Phân tích')
tab_control.pack(expand=1, fill='both')
tab2()

# Start the main event loop
root.mainloop()