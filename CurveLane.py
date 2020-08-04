import cv2
import numpy as np
from scipy.interpolate import interp1d
root = '/train/CurveLanes/Curvelanes/train'

def fit_points(x_list,y_list):

    if len(x_list) < 5:
        fit_param = np.polyfit(y_list, x_list, 1)
        y_list = np.linspace(y_list[0],y_list[-1], len(y_list))
        x_list = fit_param[0]*y_list + fit_param[1]
        
    else:
        fit_param = np.polyfit(y_list, x_list, 2)
        y_list = np.linspace(y_list[0],y_list[-1], len(y_list))
        x_list = fit_param[0]*y_list**2 + fit_param[1]*y_list + fit_param[2]
    
    return y_list,x_list

def interpolate(y_list,x_list,h):

    new_y = []
    new_x = []
    for i in range(len(y_list)):
        if y_list[i] not in new_y and y_list[i]<h:
            new_y.append(y_list[i])
            new_x.append(x_list[i])
    y_list = new_y 
    x_list = new_x

    if len(y_list)>4:
        li = interp1d(y_list,x_list,kind='cubic')
    else:
        li = interp1d(y_list,x_list)

    y_list = np.linspace(y_list[-1],y_list[0], int(y_list[0] - y_list[-1]))
    x_list = li(y_list)
    return y_list,x_list

def plot(image_p,label):
    image = cv2.imread(image_p)
    instance = np.zeros([image.shape[0], image.shape[1]], np.uint8)
    color = 0
    for line in label:
        x_list = []
        y_list = []
        for c in line:
            x0 = float(c['x'])
            y0 = float(c['y'])
            x_list.append(x0)
            y_list.append(y0)

        y_list,x_list = interpolate(y_list,x_list,image.shape[0])
        all_height = y_list[-1] - y_list[0]
        for idx in range(len(y_list)-1):
            x0 = x_list[idx]
            y0 = y_list[idx]
            x1 = x_list[idx+1]
            y1 = y_list[idx+1]
            line_width = int((2 + (y1 - y_list[0])*15.0/all_height)*image.shape[1]/1280)
            cv2.line(instance,(int(x0),int(y0)),(int(x1),int(y1)),color=color,thickness=line_width)
            cv2.line(image,(int(x0),int(y0)),(int(x1),int(y1)),color=(255,0,0),thickness=line_width)
        color += 1
    bn = os.path.basename(image_p)
    cv2.imwrite('result/{}'.format(bn),image)
    cv2.imwrite('label/{}'.format(bn.replace('.jpg','.png')),instance)

            

import os
import cv2
import json
for idx,f in enumerate(os.listdir(os.path.join(root,'images'))):
    image_p = os.path.join(root,'images',f)
    json_p = os.path.join(root,'labels',f.replace('.jpg','.lines.json'))
    with open(json_p) as f:
        label = json.load(f)

    plot(image_p,label['Lines'])
    print(idx)
    # if idx == 100:
    #     break
