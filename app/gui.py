
import base64

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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) 

from utils.tools import *
from configs.ObjectConfig import glo_config 
from operations.cv_operation import *
from operations.glofeatures_operation import *
from operations import model_operation
from frontend import st_label_studio
from utils.vl_model_api import init_openai_client, query_model, query_model_with_image

def load_webui_models(glo_config):
    # 加载webui需要的模型
    seg_model = model_operation.load_ResUNet_model(glo_config["seg_model_path"], 'cpu')

    edd_detector = model_operation.load_edd_detector_model(glo_config["edd_model_path"], glo_config["edd_model_config"], 'cpu')

    podo_cls_model = model_operation.load_podo_cls_model(glo_config["podo_cls_model_path"], 'cpu')

    return seg_model, edd_detector, podo_cls_model

def manege_image_properties(st, c1_col2_con):

    # 使用session_state来存储输入字段的值
    if 'image_id' not in st.session_state:
        st.session_state['image_id'] = ""
    if 'magnification' not in st.session_state:
        st.session_state['magnification'] = ""
    if 'ruler_tag' not in st.session_state:
        st.session_state['ruler_tag'] = 0.0
    if 'ruler_pixlen' not in st.session_state:
        st.session_state['ruler_pixlen'] = 0.0

    
    image_id = c1_col2_con.text_input("Image_ID", value=st.session_state['image_id'])
    magnification = c1_col2_con.text_input("Magnification", value=st.session_state['magnification'])
    ruler_tag = c1_col2_con.number_input("Ruler_Tag, e.g. 3000 nm", value=st.session_state['ruler_tag'])
    ruler_pixlen = c1_col2_con.number_input("Ruler_PixLen, e.g. 500", value=st.session_state['ruler_pixlen'])
    submit = c1_col2_con.button("Submit")

    if submit:
        print('更新session_state')
        st.session_state['image_id'] = image_id
        st.session_state['magnification'] = magnification
        st.session_state['ruler_tag'] = ruler_tag
        st.session_state['ruler_pixlen'] = ruler_pixlen
    
    return st.session_state, c1_col2_con

def load_example_img(st, webui_state):

    example_img_path = "app/frontend/images/examples/test-151441.jpg"
    webui_state.task['data']['image'] = ['images/examples/test-151441.jpg']

    image = Image.open(example_img_path)
    c1_col1_con = st.container(border=True)
    c1_col1_con.header("Example  image")
    c1_col1_con.image(image, width=700)
    
    st.session_state['image_id'] = 'test-151441.jpg'
    st.session_state['magnification'] = '2.5KX'
    st.session_state['ruler_tag'] = 5000
    st.session_state['ruler_pixlen'] = 500

    return image, st.session_state, c1_col1_con, webui_state

def main(glo_config, webui_state):
    # 侧边栏
    with st.sidebar:
        # 标题栏
        containertitle = st.container(border=False)
        containertitle.markdown('<div style="text-align: center; font-weight: bold; font-size: 24px;">Glo-DMU: Glomeruli TEM Image Analysis Tools</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

        # 图像上传区
        containerstep1 = st.container(border=True)
        containerstep1.header("Step 1: Upload")
        uploaded_files = containerstep1.file_uploader('',accept_multiple_files=True, type=["jpg", "jpeg", "png"])
        st.markdown('<br>', unsafe_allow_html=True)

        #任务特异模型调用区
        containerstep2 = st.container(border=True)
        containerstep2.header("Step 2: Task-specific Models")
        run_model_button = containerstep2.button("Run Model")
        st.markdown('<br>', unsafe_allow_html=True)

        #基础模型调用区
        containerstep3 = st.container(border=True)
        containerstep3.header("Step 3: VL-Foundation Model")
        generate_report_button = containerstep3.button("Generate TEM report")
        st.markdown('<br>', unsafe_allow_html=True)

    # 图像处理区
    c1 = st.container(border=True) 
    c2 = st.container(border=True) 
    c3 = st.container(border=True)
    c4 = st.container(border=True) # 视觉模型问答区

    with c1:
        c1_col1, c1_col2 = st.columns([0.6,0.4]) 
        # 图像展示区
        with c1_col1:
            if uploaded_files:

                folder_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                folder_path_prefix = f"app/frontend/images/upload_imgs/{folder_stamp}"
                os.makedirs(folder_path_prefix, exist_ok=True)

                for uploaded_file in uploaded_files:
                    
                    # 将上传的文件转换为图像
                    image = Image.open(uploaded_file)
                    
                    # 将图像保存到临时文件
                    img_array = np.array(image)
                    temp_file_path = os.path.join(folder_path_prefix, uploaded_file.name)
                    cv2.imwrite(temp_file_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


                webui_state.task['data']['image'] = [os.path.join('images/upload_imgs', folder_stamp, x) for x in os.listdir(folder_path_prefix)] #只加载png

                print(webui_state.task['data']['image'])
                # st_label_studio(description, config_x, interfaces, user, task)

                c1_col1_con = st.container(border=True)
                c1_col1_con.markdown("Uploaded  image")
                c1_col1_con.image(image, width=600)

            else:
                # 不上传图片，则加载默认图片
                image, st.session_state, c1_col1_con, webui_state = load_example_img(st, webui_state)


        # 图像属性输入区
        with c1_col2:
            c1_col2_con = st.container(border=True)
            c1_col2_con.header("Image Properties")
            st.session_state, c1_col2_con = manege_image_properties(st, c1_col2_con) # 管理图像属性


    if run_model_button:
        
        for file_path in webui_state.task['data']['image']:
            file_path = os.path.join('app/frontend/', file_path)
            print(file_path)
            image = Image.open(file_path)
            img_array = np.array(image)
            width, height= image.size
            img_info_dict = {'Image_Shape':str((height, width,  3)), 'pixel_sample_stride': 40, 'Per_Pixel_Dist':10, 'pixel_crop_window':256}


            # 模型推理
            label = model_operation.predict_seg_mask(seg_model, file_path, 'cpu')
            gbm_mask = (label==1).astype(np.uint8) * 255
            # gbm_mask, label = Predictor.predict_img_mask(seg_model, file_path, config["device"])
            GBM_morp_list = get_mask_morphology(gbm_mask)
            GBM_mask, boundary_map, gbm_centerlines, queued_lines = GBM_morp_list
            sampling_points, sampling_points_orginal = sampling_on_grid(img_info_dict, GBM_morp_list)
            gfb_patches_list, patches_coords_list = get_gfb_patches(img_info_dict, img_array, sampling_points_orginal,) 

            # GBM
            normal_k_list = fit_centerline(gbm_centerlines, sampling_points, glo_config["pixel_fit_fov"])  
            points_on_boundary = find_points_boundary(normal_k_list, boundary_map)
            img_info_dict = match_img_label_size(img_info_dict, points_on_boundary, eval(img_info_dict['Image_Shape']), GBM_mask.shape)
            img_dist = plot_thcikness(img_info_dict['gbm_thickness_result'], img_array)

            # FPE
            pred_list= model_operation.predict_podo_patches_cls(podo_cls_model, gfb_patches_list, glo_config) 
            img_podo = plot_podo_fusion_webui(img_array, list(pred_list)[0].tolist(), patches_coords_list)

            # EDD
            # edd_bbox = Predictor.predict_edd_position(edd_detector, file_path)
            edd_bbox = model_operation.predict_edd(edd_detector,  file_path)

            label = cv2.resize(label,(width,height),interpolation=cv2.INTER_NEAREST)

            edd_bboxes = edd_bbox.bboxes.cpu().numpy()
            edd_scores = edd_bbox.scores.cpu().numpy()

            edd_info = match_seg_det_output(label, edd_bboxes)
            edd_info = edd_info[edd_scores >= glo_config['edd_threshold']]
            box_num_dict = calculate_box_kind(edd_info)

            edd_position_df = pd.DataFrame(box_num_dict, index=[0])
  
        

            img_edd = plot_edd_location_webui(img_array, edd_info, label, glo_config)


            
            webui_state.task = edd_cvt_lsf(file_path,edd_bbox,webui_state.task, cv2.resize(gbm_mask,(width,height)))
            # # 重新渲染界面以显示新的预测结果   
            with c2: 
                c2.header("Annotation GUI")
                # st_label_studio(description, config_x, interfaces, user, webui_state.task)
                call_lsf(description, config_x, interfaces, user, webui_state.task,)

            with c3:
                c3.header("Visualization of ultrastructural features")
                st.markdown("---")
                
                # 创建三列布局
                col1, col2, col3 = st.columns(3)
                
                # 在每列中显示对应的图像
                with col1:
                    st.markdown("**GBM Thickness**")
                    st.image(img_dist, width=400)
                    
                    st.markdown("**GBM Thickness Data**")
                    st.json(img_info_dict)

                
                with col2:
                    st.markdown("**FPE Degree**")
                    st.image(img_podo, width=400)
                    st.markdown("**FPE Degree Data**")
                    
                    st.dataframe(pd.DataFrame(pred_list[0]))
                    
                    st.dataframe(pd.DataFrame(patches_coords_list))
                
                with col3:
                    st.markdown("**EDD Location**")
                    st.image(img_edd, width=400)
                    st.markdown("**EDD Location Data**")
                    st.dataframe(pd.DataFrame(patches_coords_list))
                    st.json(edd_info)
                    st.json(box_num_dict)

            
    with c4:
        c4.header("TEM Visual Question Answering")
        st.markdown("---")
        
        # 初始化问答历史记录
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # 创建两列布局：左侧显示当前图片，右侧是问答区域
        qa_col1, qa_col2 = st.columns([0.6, 0.4])
        
        with qa_col1:
            st.image(image, caption="Current Image", width=700)
        
        with qa_col2:
            # 问答输入区域
            user_question = st.text_input("Ask a question about the image:", key="user_question")
            send_button = st.button("Send", key="send_button")
            if generate_report_button:
                user_question = "请观察这张肾活检透射电镜图像，推测这位病人的毛细血管、基底膜、足突、电子致密物沉积、系膜区的状态"
                send_button = True

            if send_button and user_question:
                try:
                    # 初始化客户端（如果还没有初始化）
                    if 'vl_client' not in st.session_state:
                        st.session_state.vl_client = init_openai_client()
                    
                    # 获取当前图片路径
                    current_image_path = "/public/zhangzhentai/code/glo_dmu/app/frontend/images/examples/test-151441.jpg"
                    
                    # 使用query_model_with_image获取响应
                    response = query_model_with_image(
                        st.session_state.vl_client,
                        current_image_path,
                        user_question
                    )
                    
                    # 添加到历史记录
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response
                    })
                    
                except Exception as e:
                    st.error(f"Error during model inference: {str(e)}")
            
            # 显示历史对话
            st.markdown("### Chat History")
            for chat in st.session_state.chat_history:
                st.markdown(f"##### **Q:** {chat['question']}")
                st.markdown(f"##### **A:** {chat['answer']}")
                st.markdown("---")

    # # 当点击submit按钮时执行
    # if submit:
    #     # 创建一个字典来保存输入的值
    #     data = {
    #         "Image_ID": image_id,
    #         # "Patient_ID": patient_id,
    #         "Magnification": magnification,
    #         "Ruler_Tag": ruler_tag,
    #         "Ruler_PixLen": ruler_pixlen
    #     }
        
    #     # 将字典转换为JSON格式
    #     json_data = json.dumps(data, indent=4)
        
    #     # 保存为JSON文件
    #     with open("image_properties.json", "w") as json_file:
    #         json_file.write(json_data)
        
    #     # 显示保存成功的消息
    #     st.success("Image properties have been saved to image_properties.json")


    

if __name__ == '__main__':

    st.set_page_config(layout='wide')
    webui_state = st.session_state

    seg_model, edd_detector, podo_cls_model = load_webui_models(glo_config)

    # GUI模块的初始化
    description, config_x, interfaces, user, webui_state.task = get_config()

    # 主界面
    main(glo_config, webui_state)
