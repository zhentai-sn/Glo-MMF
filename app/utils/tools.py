
import streamlit as st
from PIL import Image
import json
import io
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import glob
import os
from datetime import datetime
import json
import uuid
from typing import Tuple

from app.frontend import st_label_studio


def generate_unique_id():
    return str(uuid.uuid4())


def get_config():
    label_studio_app_config = json.loads(io.open('app/weiui_config.json', 'r', encoding='utf8').read())

    description = label_studio_app_config['description']
    interfaces = label_studio_app_config['interfaces']
    user = label_studio_app_config['user']
    task = label_studio_app_config['task']
    config_path = label_studio_app_config['config']

    # 读取XML文件
    tree = ET.parse(config_path)
    config_x = tree.getroot()
    config_x = ET.tostring(config_x, encoding='unicode')
    

    return description, config_x, interfaces, user, task

def edd_cvt_lsf(temp_file_path, edd_position, task,gbm_mask):
    img = cv2.imread(temp_file_path)
    img_h, img_w, _ = img.shape
    edd_threshold = 0.9
    edd_scores = edd_position.scores[edd_position.scores >= edd_threshold]
    edd_bboxes = edd_position.bboxes[edd_position.scores >= edd_threshold]
    edd_labels = edd_position.labels[edd_position.scores >= edd_threshold]

    # 坐标归一化
    edd_bboxes[:, [0, 2]] /= img_w
    edd_bboxes[:, [1, 3]] /= img_h
    edd_bboxes *=100
    edd_bboxes = edd_bboxes.cpu().numpy().astype(np.float64)

    # 坐标格式转化
    x_list,  y_list= edd_bboxes[:,0], edd_bboxes[:,1]
    height_list  =  edd_bboxes[:, 2] - edd_bboxes[:, 0]  # Calculate width
    width_list =  edd_bboxes[:, 3] - edd_bboxes[:, 1]  # Calculate height

    # 更新task
    for x,y,h,w in zip(x_list, y_list, height_list, width_list):

        time_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")+ str(datetime.now().microsecond)
        task['annotations'][0]['result'].append({   
                            "original_width": 100,
                            "original_height": 100,
                            "from_name": "tag",
                            "id": f"{time_id}",
                            "source": "$image",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "rectanglelabels": ["EDD"],
                                "rotation": 0,

                            }
                            
                        })
    rle = mask2rle(gbm_mask) 
    task['annotations'][0]['result'].append(
        {
            "id":"123",
            "from_name":"labels",
            "to_name":"image",
            "type":"brushlabels",
            "origin":"manual",
            "value":{
                "format": "rle",
                "rle":rle,
                "brushlabels": ["GBM"]
            }
        }
    )
  
    return task
    
def bits2byte(arr_str, n=8):
    """Convert bits back to byte

    :param arr_str:  string with the bit array
    :type arr_str: str
    :param n: number of bits to separate the arr string into
    :type n: int
    :return rle:
    :type rle: list
    """
    rle = []
    numbers = [arr_str[i : i + n] for i in range(0, len(arr_str), n)]
    for i in numbers:
        rle.append(int(i, 2))
    return rle

def base_rle_encode(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]

def encode_rle(arr, wordsize=8, rle_sizes=[3, 4, 8, 16]):


    # 用32位设置数组的长度
    num = len(arr)
    numbits = f'{num:032b}'

    # 设置wordsize的位数
    wordsizebits = f'{wordsize - 1:05b}'

    # 将rle_sizes转换为位数
    rle_bits = ''.join([f'{x - 1:04b}' for x in rle_sizes])

    # 将这些部分组合成基础字符串
    base_str = numbits + wordsizebits + rle_bits

    # 开始创建RLE位字符串
    out_str = ''
    for length_reeks, p, value in zip(*base_rle_encode(arr)):
        # TODO: 这部分可以优化，但目前功能正常
        if length_reeks == 1:
            # 表示该数值的长度为1，用第一个0表示
            out_str += '0'
            # 用00表示rle_sizes中的索引
            out_str += '00'
            # rle_size值为0，表示单个数字
            out_str += '000'
            # 将数值转换为8位字符串
            out_str += f'{value:08b}'
            state = 'single_val'

        elif length_reeks > 1:
            state = 'series'
            # rle size = 3
            if length_reeks <= 8:
                # 用1表示开始一个系列
                out_str += '1'
                # rle_sizes数组中的索引
                out_str += '00'
                # 将系列长度转换为位
                out_str += f'{length_reeks - 1:03b}'
                # 将数值转换为8位字符串
                out_str += f'{value:08b}'

            # rle size = 4
            elif 8 < length_reeks <= 16:
                # 用1表示开始一个系列
                out_str += '1'
                out_str += '01'
                # 将系列长度转换为位
                out_str += f'{length_reeks - 1:04b}'
                # 将数值转换为8位字符串
                out_str += f'{value:08b}'

            # rle size = 8
            elif 16 < length_reeks <= 256:
                # 用1表示开始一个系列
                out_str += '1'
                out_str += '10'
                # 将系列长度转换为位
                out_str += f'{length_reeks - 1:08b}'
                # 将数值转换为8位字符串
                out_str += f'{value:08b}'

            # rle size = 16或更长
            else:
                length_temp = length_reeks
                while length_temp > 2**16:
                    # 用1表示开始一个系列
                    out_str += '1'
                    out_str += '11'
                    out_str += f'{2**16 - 1:016b}'
                    out_str += f'{value:08b}'
                    length_temp -= 2**16

                # 用1表示开始一个系列
                out_str += '1'
                out_str += '11'
                # 将系列长度转换为位
                out_str += f'{length_temp - 1:016b}'
                # 将数值转换为8位字符串
                out_str += f'{value:08b}'

    # 确保最终字符串长度为8的倍数，不足时在末尾补0
    nzfill = 8 - len(base_str + out_str) % 8
    total_str = base_str + out_str
    total_str = total_str + nzfill * '0'

    # 将位字符串转换为字节
    rle = bits2byte(total_str)

    return rle

def mask2rle(mask):
    """Convert mask to RLE

    :param mask: uint8 or int np.array mask with len(shape) == 2 like grayscale image
    :return: list of ints in RLE format
    """
    assert len(mask.shape) == 2, 'mask must be 2D np.array'
    assert mask.dtype == np.uint8 or mask.dtype == int, 'mask must be uint8 or int'

    array = mask.ravel()
    array = np.repeat(array, 4)  # must be 4 channels
    rle = encode_rle(array)
    return rle

def set_page():
    pass


def call_lsf(description, config_x, interfaces, user, state,):
    st_label_studio(description, config_x, interfaces, user, state,)

# def plot_thcikness(img_dist_df, img):

#         # gbm_centerlines = cv2.resize(gbm_centerlines, (img.shape[1], img.shape[0], ))
#         # boundary_map = cv2.resize(boundary_map, (img.shape[1], img.shape[0], ))
#         # bk_mat = np.zeros_like(gbm_centerlines)
#         # draw_mask = np.stack([gbm_centerlines, bk_mat, boundary_map], axis=-1)

#         img_dist_df['is_suitabel_measured'] = 1 # TODO 临时

#         GBM_MEAN_MODIFY_FACTOR = np.pi/4
#         line_width=8
#         measured_color = np.array((255,102,0)) # 橙色 /255
#         unmeasured_color = np.array((85,34,0)) # 棕色 /255

#         result_img = img.copy()
#         for index, row in img_dist_df.iterrows():
#             if abs(row['corrcoef'])<=0.6:
#                 continue

#             p = row['measured_points']
#             x = [x[0] for x in p]  # 行坐标
#             y = [y[1] for y in p]  # 列坐标

#             if row['is_suitabel_measured']==1:
#                 cv2.line(result_img, tuple(p[0][::-1]), tuple(p[1][::-1]), measured_color.tolist(), line_width)
#                 cv2.putText(result_img, str(int(np.round(row['dist']))), (max(y)+ 10, max(x)+ 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)

#             else:
#                 cv2.line(result_img, tuple(p[0][::-1]), tuple(p[1][::-1]), unmeasured_color.tolist(), line_width)
    
#         measured_dist_list = img_dist_df.loc[img_dist_df['is_suitabel_measured']==1, 'dist']
#         th_mean_std = f'{np.round(np.mean(measured_dist_list)*GBM_MEAN_MODIFY_FACTOR, 2)}±{np.round(np.std(measured_dist_list)*GBM_MEAN_MODIFY_FACTOR,2)}'
#         # cv2.putText(result_img, th_mean_std,(50, 50,),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 5, cv2.LINE_AA)
#         # result_img = cv2.addWeighted(result_img, 1, draw_mask, 0.5, 0)

#         return result_img

def plot_thcikness(gbm_thickness_result: list, img: np.ndarray) -> np.ndarray:
    """绘制GBM厚度测量结果

    Args:
        gbm_thickness_result (list): GBM厚度测量结果列表，每个元素为字典，包含测量点、像素长度、相关系数和距离
        img (np.ndarray): 原始图像

    Returns:
        np.ndarray: 标注后的图像
    """
    GBM_MEAN_MODIFY_FACTOR = np.pi/4
    line_width = 8
    measured_color = np.array((255,102,0))  # 橙色
    unmeasured_color = np.array((85,34,0))  # 棕色

    result_img = img.copy()
    valid_measurements = []

    # 遍历所有测量结果
    for measure in gbm_thickness_result:
        if abs(measure['corrcoef']) <= 0.6:
            continue

        p = measure['measured_points']
        x = [x[0] for x in p]  # 行坐标
        y = [y[1] for y in p]  # 列坐标
        
        # 绘制测量线
        cv2.line(result_img, tuple(p[0][::-1]), tuple(p[1][::-1]), measured_color.tolist(), line_width)
        cv2.putText(result_img, str(int(np.round(measure['dist']))), 
                   (max(y) + 10, max(x) + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
        
        valid_measurements.append(measure['dist'])

    # 计算平均值和标准差
    if valid_measurements:
        mean_dist = np.mean(valid_measurements) * GBM_MEAN_MODIFY_FACTOR
        std_dist = np.std(valid_measurements) * GBM_MEAN_MODIFY_FACTOR
        th_mean_std = f'{np.round(mean_dist, 2)}±{np.round(std_dist, 2)}'
        # 如果需要显示统计值，取消下面的注释
        # cv2.putText(result_img, th_mean_std, (50, 50), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 5, cv2.LINE_AA)

    return result_img

def plot_podo_fusion_webui(img: np.ndarray, pred_list, coordinate_list, save_path=''):
        alpha = 0.8
        img_podo = img.copy()
        img_rect = np.zeros_like(img)


        if pred_list is None:
            return
        
        for pred, coordinate in zip(pred_list, coordinate_list):
            xy, win_width, = coordinate
            halfwin = win_width//2
            y1, x1 =xy # y1为行坐标， X1为列坐标
            
            if pred==1: 
                cv2.rectangle(img_rect, (x1-halfwin, y1-halfwin), (x1 + halfwin, y1 + halfwin), (0,0,255), thickness=-1)

            else:
                cv2.rectangle(img_rect, (x1-halfwin, y1-halfwin), (x1 + halfwin, y1 + halfwin), (0,255,0), thickness=-1)

        img_podo = cv2.addWeighted(img_podo, alpha, img_rect, 1 - alpha, 0)


        return img_podo

def plot_edd_location_webui(img, edd_info, label, config,):

    # default_color = (255, 255, 0)

    color_map = config["ultrastructure_colors"]

    img_H, img_W, _ =  img.shape
    label_H, label_W =  label.shape

    # paint color
    label_color = np.zeros((label_H, label_W, 3), dtype=np.uint8)
    for tag, color in color_map.items():     
        label_color[label == int(tag)] = color

    overlay = img

        

    # 绘制检测方框
    for box in edd_info:
        x1, y1, x2, y2 = box[:4].astype(int)
        category = box[4:].astype(int)

        # 根据类别选择颜色
        color = color_map[str(int(category))]
        # color = default_color
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
        # cv2.putText(overlay, str(np.round(p, 3)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 计数信息 # 0:unsure, 1:GBM 2:E, 3:podo, 4:MC 在mGBM中的电子致密物也算作基底膜内沉积
    a = sum(edd_info[:,4]==0) 
    b = sum(edd_info[:,4]==1) + sum(edd_info[:,4]==4)
    c = sum(edd_info[:,4]==2)
    d = sum(edd_info[:,4]==3)
    e = sum(edd_info[:,4]>4)
    text =f"gbm:{b},E:{c},podo:{d},unsure:{a},Mc:{e}"
    text_position = (img_W - 800, img_H - 30)
    cv2.putText(overlay, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4)    

    # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay = cv2.resize(overlay,(overlay.shape[1] // 2, overlay.shape[0] // 2))
    # overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay

def convert_uploaded_file_to_image(uploaded_file, folder_path_prefix: str) -> Tuple[str, int, int]:
    """
    将上传的文件转换为图像，并保存到临时文件，同时返回临时文件路径、图像宽度和高度。

    :param uploaded_file: 上传的文件对象，通常来自文件上传接口。
    :param folder_path_prefix: 临时文件保存的文件夹路径前缀。
    :return: 包含临时文件路径、图像宽度和高度的元组。
    """
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    temp_file_path = os.path.join(folder_path_prefix, uploaded_file.name)
    cv2.imwrite(temp_file_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    width, height = image.size

    return image, temp_file_path, width, height