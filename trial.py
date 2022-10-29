
from matplotlib import pyplot as plt
import os
import streamlit as st
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from pathlib import Path
from object_detection.utils import dataset_util, label_map_util
#import label_map_util
from object_detection.protos import string_int_label_map_pb2
from object_detection.protos import pipeline_pb2
from object_detection.utils import visualization_utils as vis_util
from tensorflow.python.util import compat
from tensorflow.core.protobuf import saved_model_pb2

def reconstruct(pb_path):
    if not os.path.isfile(pb_path):
        print("Error: %s not found" % pb_path)

    print("Reconstructing Tensorflow model")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("Success!")
    return detection_graph

def image2np(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)

def image2tensor(image):
    npim = image2np(image)
    return np.expand_dims(npim, axis=0)

def detect(detection_graph, test_image_path):
    with detection_graph.as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.01)#per_process_gpu_memory_fraction=0.01
        with tf.compat.v1.Session(graph=detection_graph,config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image = Image.open(test_image_path)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image2tensor(image)}
            )

            npim = image2np(image)
            vis_util.visualize_boxes_and_labels_on_image_array(
                npim,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=15)
            plt.figure(figsize=(12, 8))
            plt.imshow(npim)
            plt.show()

PB_PATH = "model/ssd_mobilenet_v2_taco_2018_03_29.pb"
LABEL_PATH = 'data/graph.pbtxt'
NCLASSES = 60
detection_graph = reconstruct(PB_PATH)

label_map = label_map_util.load_labelmap('data/labelmap.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NCLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


#st.title('Recyclables are all we need')
st.markdown("<h1 style='text-align: center;'>Recyclables Are All We Need</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>CS604 - Deep Learning for Visual Recognition, Group 6 - MobilenetV2 MVP</h3>", unsafe_allow_html=True)
st.empty()
st.caption("Members: Huang Jing, Ruipeng, Shengming, Wei Song, Zhang Ge")

# picture = st.camera_input("Take a picture")

#st.subheader("Image")
st.set_option('deprecation.showPyplotGlobalUse', False)
image_file = st.file_uploader("Please add images of recyclable objects you have trouble identifying:", type=["png","jpg","jpeg"])
if image_file:
    # st.image(image_file)
    with open(os.path.join("demo.jpg"),"wb") as f:
        f.write((image_file).getbuffer())
        saved = st.success("File Saved")
    
    if saved:
        detect(detection_graph,"demo.jpg")
        # plt.show()
        st.pyplot(plt.show())