#!/usr/bin/env python
from flask import Flask, render_template, Response, request, redirect
from waitress import serve
import io
import cv2
import tensorflow as tf
import align.detect_face as detect_face
import facenet.facenet as facenet
import cv2
import time
import numpy as np
import glob
import pickle
import collections
import os
# from pytube import YouTube
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

project_root_folder = os.path.abspath(os.path.join(os.getcwd(), "../.."))
classifier_path = os.path.join(project_root_folder, 'models/kol.pkl')
model_path = os.path.join(project_root_folder, 'models/20180402-114759/')
with open(classifier_path, 'rb') as infile:
    (model, class_names) = pickle.load(infile)
    print("Loaded classifier file")

@app.route('/')
def index():
    """Video streaming home page."""
    global file_path
    file_path = request.args.get('file_path')
    return render_template('index.html')

def mean(alist):
    sum = 0
    for item in alist:
        sum += item
    return sum/len(alist)

def gen(static_path):
    """Video streaming generator function."""
    # tf.reset_default_graph()
    try:
        while True:
            minsize = 20
            threshold = [0.6, 0.7, 0.7]
            factor = 0.709
            # image_size = 182
            input_image_size = 160

            video_speedup = 5
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            people_detected     = set()
            person_average_prob = dict()
            people_aver_prob    = []
            person_detected     = collections.Counter()

            global graph
            tf.reset_default_graph()
            with graph.as_default():
                # pnet, rnet, onet = detect_face.create_mtcnn(sess, project_root_folder + "\\src\\align")
                video_capture_path = static_path
                if not os.path.isfile(video_capture_path):
                    print('Video not found at path ' + video_capture_path)
                video_capture = cv2.VideoCapture(video_capture_path)
                width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) 
                height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                total_frames_passed = 0
                while True:
                    try:
                        ret, frame = video_capture.read()
                    except Exception as e:
                        break

                    if video_speedup:
                        total_frames_passed += 1
                        if total_frames_passed % video_speedup != 0:
                            continue
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    if bounding_boxes == '0':
                        print("Note! Do not detect one face!!!!!!!!")
                        break
                    else:
                        faces_found = bounding_boxes.shape[0]

                        if faces_found > 0:
                            det = bounding_boxes[:, 0:4]
                            bb = np.zeros((faces_found, 4), dtype=np.int32)
                            for i in range(faces_found):
                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]
                                # inner exception
                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                    print('face is inner of range!')
                                    continue

                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)

                                scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                best_name = class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                                if best_class_probabilities > 0.09:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    display = "%s % 5.4f" %(best_name, best_class_probabilities)
                                    # 人脸框: 检测到的对象 | 当前预测可能性
                                    cv2.putText(frame, display, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                                    person_detected[best_name] += 1
                                    person_average_prob.setdefault(best_name,[]).append(best_class_probabilities)

                        # 第一个text行: 出现次数最多的KOL: identity | 当前预测可能性
                        for name, count in person_detected.most_common(1):
                            display = "Top KOL: %s % 5.4f" %(name, person_average_prob[name][-1])
                            cv2.putText(frame, display, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2, lineType=2)
                        idx = 0
                        currentYIndex = 40
                        # 第二个text行: 按照次数依次排列KOL: identity | 平均可能性
                        for name, count in person_detected.most_common(5):
                            aver_prob = mean(person_average_prob[name]) 
                            display = "%s: % 5.4f" %(name, aver_prob[0])
                            cv2.putText(frame, display, (20, currentYIndex + 10 + 20 * idx), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), thickness=1, lineType=2)
                            idx += 1  

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
                        io_buf = io.BytesIO(image_buffer)
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')
                video_capture.release()
                cv2.destroyAllWindows()
                break
    except:
        pass

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(file_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
# tf.reset_default_graph()
    start_time  = time.time()
    with tf.Graph().as_default():
        graph = tf.get_default_graph()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 指定使用第一块GPU
        config                                              =  tf.ConfigProto()
        config.allow_soft_placement                         = True
        config.gpu_options.per_process_gpu_memory_fraction  = 0.7
        config.gpu_options.allow_growth                     = True
        with tf.Session(config=config) as sess:
            facenet.load_model(model_path)
            pnet, rnet, onet = detect_face.create_mtcnn(sess, project_root_folder + "\\src\\align")
            images_placeholder      = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings              = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print("Server runing at locathost:7000!")
            spend = time.time() - start_time
            print(str(spend)+' seconds | ' + str(spend/60) + ' minutes') # 214 seconds | 3.57 minutes
            serve(app=app, host='0.0.0.0', port=7000)
            # serve(app=app, host='0.0.0.0', port=5000, debug=True, threaded=True)
            # app.run(host='0.0.0.0', debug=True, threaded=True)