#!/usr/bin/env python
from flask import Flask, render_template, Response, request, redirect
import io
import cv2
import tensorflow as tf
import align.detect_face
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
# vc = cv2.VideoCapture(0)

project_root_folder = os.path.abspath(os.path.join(os.getcwd(), "../.."))
classifier_path = os.path.join(project_root_folder, 'models/kol.pkl')
model_path = os.path.join(project_root_folder, 'models/20180402-114759/')

@app.route('/')
def index():
    """Video streaming home page."""
    # global file_path
    # file_path = request.args.get('file_path')
    # global classifier_path
    # classifier_path = request.args.get('classifier_path')
    # global model_path
    # model_path = request.args.get('model_path')
    # global port
    # port = request.args.get('port')
    # redirect_url = "http://localhost:" + str(port) + "/uploadVideoPage"
    # return render_template('index.html', redirect_url=redirect_url)
    global file_path
    file_path = request.args.get('file_path')
    return render_template('index.html')

def mean(alist):
    sum = 0
    for item in alist:
        sum += item
    return sum/len(alist)

def gen(file_path): # def gen(file_path, stop):
    """Video streaming generator function."""
    try:
        while True:
            # if stop == 1:
            #     break
            video_speedup = 5

            minsize = 20
            threshold = [0.6, 0.7, 0.7]
            factor = 0.709
            image_size = 182
            input_image_size = 160
            
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            video_project_folder = os.path.abspath(os.path.dirname(__file__))
            with open(classifier_path, 'rb') as f:
                (model, class_names) = pickle.load(f)
                print("Loaded classifier file")
            people_detected     = set()
            person_average_prob = dict()
            people_aver_prob    = []
            person_detected     = collections.Counter()
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, video_project_folder + "\\align")
                    facenet_model_path = model_path
                    facenet.load_model(facenet_model_path)

                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    embedding_size = embeddings.get_shape()[1]

                    video_capture_path = file_path   
                    # youtube_video_url = 'https://www.youtube.com/watch?v=WW1VmFi_18Y'
                    if not os.path.isfile(video_capture_path):
                        print('Video not found at path ' + video_capture_path + '. Commencing download from YouTube')
                        # YouTube(youtube_video_url).streams.first().download(output_path =video_path, filename=video_name)
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
                        bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)

                        if bounding_boxes == '0':
                            print("Note! Do not detect one face!!!!!!!!")
                            break
                        else:
                            faces_found = bounding_boxes.shape[0]

                            start_time  = time.time()
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

                                    if best_class_probabilities > 0.09 and best_class_probabilities <= 0.7:
                                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                                        text_x = bb[i][0]
                                        text_y = bb[i][3] + 20                             
                                        cv2.putText(frame, "Others", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
                                        person_detected[best_name] += 1
                                        person_average_prob.setdefault(best_name,[]).append(best_class_probabilities)       
                                    if best_class_probabilities > 0.7:
                                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                                        text_x = bb[i][0]
                                        text_y = bb[i][3] + 20
                                        # # 人脸框: 检测到的对象 | 当前预测可能性
                                        # display = "%s % 5.4f" %(best_name, best_class_probabilities)
                                        cv2.putText(frame, best_name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
                                        person_detected[best_name] += 1
                                        person_average_prob.setdefault(best_name,[]).append(best_class_probabilities)
                            # 第一个text行: 出现次数最多的KOL: identity | 当前预测可能性
                            for name, count in person_detected.most_common(1):
                                display = "Key KOL: %s %5.4f" %(name, person_average_prob[name][-1])
                                cv2.putText(frame, display, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2, lineType=2)
                            idx = 0
                            currentYIndex = 40
                            # 第二个text行: 按照次数依次排列KOL: identity | 平均可能性
                            for name, count in person_detected.most_common(5):
                                aver_prob = mean(person_average_prob[name]) 
                                # display = "%s: % 5.4f" %(name, aver_prob[0])
                                display = "%s %s %5.4f" %(len(person_average_prob[name]), name, aver_prob)
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
    # if 'stop' in request.args:
    #     global stop
    #     stop = 1 # stop = request.args['stop']
    #     return redirect("http://localhost:" + str(port) + "/uploadVideoPage")
    # else:
    #     return Response(
    #         gen(file_path, stop=0),
    #         mimetype='multipart/x-mixed-replace; boundary=frame'
    #     )
    return Response(
        gen(file_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

from waitress import serve
if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True, threaded=True)
    print("Server runing at locathost:7000!")
    serve(app=app, host='0.0.0.0', port=7000)

    

# Simply run by: python app.py (after installing flask and opencv)
# Then just open http://0.0.0.0:5000

# Example on cross-platform video streaming based on [video streaming with Flask](http://blog.miguelgrinberg.com/post/video-streaming-with-flask) article.
# flask-opencv-streaming

