import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import joblib

import numpy as np
import cv2
import onnxruntime as ort
import imutils
# import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px



def onnx_segment_membrane(input_image, threshold):
    ort_session = ort.InferenceSession('onnx_models/membrane_segmentor.onnx')
    img = Image.fromarray(np.uint8(input_image))
    resized = img.resize((256, 256), Image.NEAREST)
    img_unsqueeze = expand_dims_twice(resized)
    onnx_outputs = ort_session.run(None, {'input': img_unsqueeze.astype('float32')}) 
    binarized = 1.0 * (onnx_outputs[0][0][0] > threshold)

    resized_ret = Image.fromarray(binarized.astype(np.uint8) ).resize((356, 256), Image.NEAREST)#.convert("L")
    centroid_img = generate_centroid_image(np.array(onnx_outputs[0][0][0])) *255
    resized_centroid_img = Image.fromarray(centroid_img.astype(np.uint8)).resize((356, 256), Image.NEAREST)
    return(resized_ret, resized_centroid_img)


def generate_centroid_image(thresh):
    thresh = cv2.blur(thresh, (5,5))
    thresh = thresh.astype(np.uint8)
    centroid_image = np.zeros(thresh.shape)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centroids = []
    for c in cnts:
        try:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            # cv2.drawContours(centroid_image, [c], -1, (255, 255, 255), 2)
            cv2.circle(centroid_image, (cX, cY), 2, (1, 1, 1), -1)
            centroids.append((cX, cY))
        except:
            pass
    return(centroid_image)



def expand_dims_twice(arr):
    norm=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
    ret = np.expand_dims(np.expand_dims(norm, axis=0), axis=0)
    return(ret)



def cell_membrane_segmentation():
    selected_box2 = st.sidebar.selectbox(
    'Choose Example Input',
    ('Example_1.png','Example_2.png')
    )

    st.title('Cell Membrane Segmentation')
    instructions = """
        Segment Cell Membrane from C. elegans embryo imaging data \n
        Either upload your own image or select from the sidebar to get a preconfigured image. 
        The image you select or upload will be fed through the Deep Neural Network in real-time 
        and the output will be displayed to the screen.
        """
    st.text(instructions)
    file = st.file_uploader('Upload an image or choose an example')
    example_image = Image.open('./images/cell_membrane_segmentation_examples/'+selected_box2)
    threshold = st.sidebar.slider("Select Threshold (Applied on model output)", 0.0, 1.0, 0.1)
    col1, col2, col3 = st.beta_columns(3)

    if file:
        input = Image.open(file)
        fig1 = px.imshow(input, binary_string=True, labels=dict(x="Input Image"))
        fig1.update(layout_coloraxis_showscale=False)
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col1.plotly_chart(fig1, use_container_width=True)

    else:
        input = example_image
        fig1 = px.imshow(input, binary_string=True, labels=dict(x="Input Image"))
        fig1.update(layout_coloraxis_showscale=False)
        fig1.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col1.plotly_chart(fig1, use_container_width=True)

    pressed = st.button('Run')
    if pressed:
        st.empty()
        model_output = onnx_segment_membrane(np.array(input), threshold)

        fig2 = px.imshow(model_output[0], binary_string=True, labels=dict(x="Segmentation Map"))
        fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col2.plotly_chart(fig2, use_container_width=True)

        fig3 = px.imshow(model_output[1], binary_string=True, labels=dict(x="Centroid Map"))
        fig3.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        col3.plotly_chart(fig3, use_container_width=True)