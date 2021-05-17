import os
import io
import cv2
import time
import align
import pickle
import shutil
import random
import numpy as np
import collections
import configparser
import tensorflow as tf
from waitress import serve
from scipy import misc
import facenet.facenet as facenet
import align.detect_face as detect_face
from pypinyin import lazy_pinyin
from werkzeug.utils import secure_filename
from imutils.video import WebcamVideoStream
from flask import Flask, request, render_template, Response, redirect
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils import (
    allowed_img_set,
    allowed_video_set,
    allowed_img_file,
    allowed_video_file,
    load_and_align_data,
    create_mtcnn,
    format_dataset,
    un_zip,
    create,
    renew
)

# tf.reset_default_graph()

app                 = Flask(__name__)
app.secret_key      = os.urandom(24)
APP_ROOT            = os.path.dirname(os.path.abspath(__file__))
# uploads_path        = os.path.join(APP_ROOT, 'static')
# APP_ROOT = os.path.abspath(os.path.dirname(__file__))
process_root_folder = os.path.join(APP_ROOT, 'other-server/preprocess/')
video_root_folder   = os.path.join(APP_ROOT, 'other-server/video/')

config = configparser.ConfigParser()
config_file = os.path.join(APP_ROOT, "config.ini")
config.read(config_file)
port = config.getint('main.server', 'port')
config.set('Project', 'root_path', APP_ROOT)
config.set('KOL.dataset', 'path', os.path.join(APP_ROOT, 'datasets/KOL'))
with open(config_file, 'w') as f:
    config.write(f)

process_config_file = os.path.join(process_root_folder, "config.ini")
video_config_file = os.path.join(video_root_folder, "config.ini")
for i in [process_config_file, video_config_file]:
    shutil.copy(config_file, i)

model               = "models/20180402-114759/"
classifier_filename = "models/kol.pkl"
classifier_filename_exp = os.path.join(APP_ROOT, os.path.expanduser(classifier_filename))
model_path = os.path.join(APP_ROOT, os.path.expanduser(model))

training_zip_path       = os.path.join(APP_ROOT, "training_set/")
predict_identity_path   = os.path.join(APP_ROOT, "static/")
upload_dir              = training_zip_path
single_image_savedir    = os.path.join(predict_identity_path, "single/")
batch_image_savedir     = os.path.join(predict_identity_path, "batch/")    
similar_image_savedir   = os.path.join(predict_identity_path, "similar/")

single_image_display_path   = "static/single/"
batch_image_display_path    = "static/batch/"
similar_image_display_path  = "static/similar/"

for dir in [training_zip_path, predict_identity_path]:
    create(dir)

with open(classifier_filename_exp, 'rb') as infile:
    (model, class_names) = pickle.load(infile)

@app.route("/")
def index_page():   # select mode: video predict | single image predict | batch image predict | find similar identity
    return render_template(template_name_or_list="index.html")

@app.route("/trainModel")
def train_model(): 
    return render_template(template_name_or_list="train_model.html")

@app.route("/trainExplain")
def train_explain(): 
    return render_template(template_name_or_list="train_explain.html")

@app.route("/trainByImage", methods=['POST', 'GET'])
def train_by_image(): 
    create(upload_dir)
    if request.method == 'POST':
        if 'zip' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="No 'zip' field in POST request!"
            )
        else:
            file        = request.files['zip']                                    # <FileStorage: 'KOL.zip' ('image/jpeg')>
            filename    = secure_filename(''.join(lazy_pinyin(file.filename)))    # KOL.zip
            if filename == "":
                return render_template(
                    template_name_or_list="warning.html",
                    status="No selected file!"
                )
            else:
                file_name, file_extension = os.path.splitext(filename)
                random_key  = np.random.randint(0, high=99999)
                output_filename = file_name + '_%5d'%(random_key)
                zip_file = os.path.join(upload_dir, output_filename + file_extension) # base_name = os.path.splitext(filename)[0]
                file.save(zip_file)
                if not os.path.exists(zip_file):
                    return render_template(
                        template_name_or_list="warning.html",
                        status="File Save Fail! Please upload your zip file again!"
                    ) 
                else:
                    if 0 in format_dataset(zip_file):
                        return render_template(
                            template_name_or_list="warning.html",
                            status="Please re-organize your training set in the standard format!"
                        )
                    else:
                        un_zip(zip_file)
                        unzip_file = os.path.join(upload_dir, os.path.splitext(zip_file)[0]) # C:\Kooler\training_set/upload/automation
                        KOL_root_dir = []
                        for dir in os.listdir(unzip_file):
                            KOL_root_dir.append(dir)
                        txt_file = os.path.join(training_zip_path, output_filename, "txt")
                        create(txt_file)
                        align_clean_train_file = os.path.join(txt_file, 'align_clean_import_train.txt')
                        if not os.path.exists(align_clean_train_file):
                            os.makedirs
                        f = open(align_clean_train_file,'w')
                        align_clean_status = '0,0,0,0,' + str(unzip_file.replace('\\','/')) + ',' + str(KOL_root_dir[0])
                        f.write(align_clean_status)    
                        port = config.getint('process.server', 'port')
                        return redirect('http://localhost:' + str(port) + '?output_filename=%s'%(output_filename))    
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="POST HTTP method required!"
        )

@app.route("/trainByVideo")
def train_by_video(): 
    return render_template(template_name_or_list="train_by_video.html")

@app.route("/uploadVideoPage")
def upload_video_page(): # manually upload video
    return render_template(template_name_or_list="predict_video.html")

@app.route('/predictVideoResult', methods=['POST', 'GET'])
def predict_video_result(): # face recognition of video
    tf.reset_default_graph()
    video_savedir = os.path.join(predict_identity_path, "video/")
    renew(video_savedir)
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="No 'file' field in POST request!"
            )
        file        = request.files['file']                                    # <FileStorage: 'download.jpg' ('image/jpeg')>
        filename    = secure_filename(''.join(lazy_pinyin(file.filename)))     # download.mp4
        if filename == "":
            return render_template(
                template_name_or_list="warning.html",
                status="No selected file!"
            )
        video_save_path = os.path.join(video_savedir, filename) # 或者把filename换成'video' + os.path.splitext(filename)[-1]做测试
        file.save(video_save_path)
        if file and allowed_video_file(filename=filename, allowed_video_set=allowed_video_set):
            # 方法1: 如果想在本服务器开启视频识别sub-server，取消注释
            # return Response(
            #     gen(video_save_path),
            #     mimetype='multipart/x-mixed-replace; boundary=frame'
            # )
            # 方法2: 另外开启一个视频识别服务器，取消注释 
            # return redirect('http://localhost:7000/?file_path=%s&classifier_path=%s&model_path=%s&port=%s' %(video_save_path,classifier_filename_exp,model_path,port)) # http://localhost:7000/ (项目下放video做测试) | 'http://localhost:7000/video_feed?file_path=%s' %video_save_path
            return redirect('http://localhost:7000/?file_path=%s' %(video_save_path))
        else:
            return render_template(
                template_name_or_list='warning.html',
                status="This file is not secure file! \n Please upload legitimate video file with extension: ['flv', 'avi', 'mp4', 'mov', 'wmv']!"
            )      
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="POST HTTP method required!"
        )

def mean(alist):
    sum = 0
    for item in alist:
        sum += item
    return sum/len(alist)

def gen(video_save_path):
    """Video streaming generator function."""
    # tf.reset_default_graph()
    while True:
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        # image_size = 182
        input_image_size = 160

        video_speedup = 5
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        # with open(classifier_filename_exp, 'rb') as infile:
        #     (model, class_names) = pickle.load(infile)
        #     print("Loaded classifier file")
        people_detected     = set()
        person_average_prob = dict()
        people_aver_prob    = []
        person_detected     = collections.Counter()

        global graph
        with graph.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, APP_ROOT + "\\src\\align")
            video_capture_path = video_save_path
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

@app.route("/predictSinglePage")
def predict_single_page(): # manually upload single image file for prediction
    return render_template(template_name_or_list="predict_single.html")

@app.route('/predictSingleImage', methods=['POST', 'GET'])
def predict_single_image(): # get multiple image prediction results | upload image files via POST request and feeds them to the FaceNet model to get prediction results
    renew(single_image_savedir) 
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="No 'file' field in POST request!"
            )
        file        = request.files['file']                                    # <FileStorage: 'download.jpg' ('image/jpeg')>
        filename    = secure_filename(''.join(lazy_pinyin(file.filename)))     # download.jpg
        if filename == "":
            return render_template(
                template_name_or_list="warning.html",
                status="No selected file!"
            )
        upload_path                                 = os.path.join(single_image_savedir, filename)
        file.save(upload_path)
        image_paths                                 = []
        image_paths.append(upload_path)
        img_display_path = os.path.join(single_image_display_path, filename)
        path_nofperson_identity_similarity_timeused = []
        display                                     = path_nofperson_identity_similarity_timeused
        if file and allowed_img_file(filename=filename, allowed_img_set=allowed_img_set):
            tf.reset_default_graph()
            start_time  = time.time()
            images, count_per_image = load_and_align_data(image_paths) # count_per_image = [x] | x = 0时代表没有一个人脸被检测到            
            if count_per_image[0] != 0:
                if count_per_image[0] == "0":
                    return render_template(
                        template_name_or_list="warning.html",
                        status="The uploaded file is illegal. Please upload safe image file!"
                    )
                feed_dict               = {images_placeholder: images , phase_train_placeholder:False}
                emb                     = sess.run(embeddings, feed_dict=feed_dict)
                if images is not None: 
                        # with open(classifier_filename_exp, 'rb') as infile:
                        #     (model, class_names) = pickle.load(infile)
                    if model:
                        print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                        predictions                 = model.predict_proba(emb)
                        best_class_indices          = np.argmax(predictions, axis=1) # <class 'numpy.ndarray'> [0]
                        best_class_probabilities    = predictions[np.arange(len(best_class_indices)), best_class_indices] # [0.99692782]
                        k = 0
                        for j in range(count_per_image[0]):
                            sentence        = str(img_display_path) + "," # 要放此项目的相对路径才显示image
                            sentence        = sentence + str(count_per_image[0]) + " people detected!,"
                            # print("\npeople in image %s :" %(filename), '%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                            identity        = class_names[best_class_indices[k]]
                            probabilities   = best_class_probabilities[k]
                            k+=1
                            probabilities   = "Similarity: " + str(probabilities).split('.')[0] + '.' + str(probabilities).split('.')[1][:3]
                            spent           = str(time.time() - start_time)
                            spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                            sentence        = sentence + "Person " + str(k) + ": " + str(identity) + "," + str(probabilities) + "," + str(spent)
                            display.append(sentence)
                    else:
                        sentence        = str(img_display_path) + ","
                        sentence        = sentence + "No embedding classifier was detected!,"
                        identity        = ""
                        probabilities   = ""
                        spent           = str(time.time() - start_time)
                        spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                        sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                        display.append(sentence)
                else:
                    sentence        = str(img_display_path) + ","
                    sentence        = sentence + "No human face was detected!,"
                    identity        = ""
                    probabilities   = ""
                    spent           = str(time.time() - start_time)
                    spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                    sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                    display.append(sentence)
            else:
                sentence        = str(img_display_path) + ","
                sentence        = sentence + "No human face was detected!,"
                identity        = ""
                probabilities   = ""
                spent           = str(time.time() - start_time)
                spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                display.append(sentence)
        else:
            return render_template(
                template_name_or_list='warning.html',
                status="This file is not secure file! Please upload image file with extension png、 jpg、 jpeg!"
            ) 
        return render_template(
            template_name_or_list='predict_single_result.html',
            display=display
        )      
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="POST HTTP method required!"
        )

@app.route("/predictBatchPage")
def predict_batch_page(): # manually upload multiple image files for prediction
    return render_template(template_name_or_list="predict_batch.html")

@app.route('/predictBatchImage', methods=['POST', 'GET'])
def predict_batch_image(): # get multiple image prediction results | upload image files via POST request and feeds them to the FaceNet model to get prediction results
    renew(batch_image_savedir)    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="No 'file' field in POST request!"
            )
        files   = request.files.getlist('file')
        path_nofperson_identity_similarity_timeused = []
        display = path_nofperson_identity_similarity_timeused
        for file in files: # file: <FileStorage: '中文.jpg' ('image/jpeg')>  
            one_image = []  
            filename = secure_filename(''.join(lazy_pinyin(file.filename))) # 中文.jpg -> zhongwen.jpg
            if allowed_img_file(filename=filename, allowed_img_set=allowed_img_set):
                if filename == "":
                    return render_template(
                        template_name_or_list="warning.html",
                        status="No selected file!"
                    )
                upload_path = os.path.join(batch_image_savedir, filename)
                file.save(upload_path)
                image_paths = []
                image_paths.append(upload_path)

                img_display_path = os.path.join(batch_image_display_path, filename)
                
                tf.reset_default_graph()
                start_time  = time.time()
                images, count_per_image = load_and_align_data(image_paths)
                if count_per_image[0] != 0:
                    if count_per_image[0] == "0":
                        return render_template(
                            template_name_or_list="warning.html",
                            status="The uploaded file is illegal. Please upload safe image file!"
                        )
                    feed_dict               = {images_placeholder: images , phase_train_placeholder:False}
                    emb                     = sess.run(embeddings, feed_dict=feed_dict)
                    if images is not None: 
                        # with open(classifier_filename_exp, 'rb') as infile:
                        #     (model, class_names) = pickle.load(infile)
                            if model: 
                                print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                                predictions                 = model.predict_proba(emb)
                                best_class_indices          = np.argmax(predictions, axis=1)
                                best_class_probabilities    = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                k = 0
                                for j in range(count_per_image[0]):
                                    sentence        = str(img_display_path) + ","
                                    sentence        = sentence + str(count_per_image[0]) + " people detected!,"
                                    # print("\npeople in image %s :" %(filename), '%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                                    identity        = class_names[best_class_indices[k]]
                                    probabilities   = best_class_probabilities[k]
                                    k+=1
                                    probabilities   = "Similarity: " + str(probabilities).split('.')[0] + '.' + str(probabilities).split('.')[1][:3]
                                    spent           = str(time.time() - start_time)
                                    spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                                    sentence        = sentence + "Person " + str(k) + ": " + str(identity) + "," + str(probabilities) + "," + str(spent)
                                    one_image.append(sentence)
                            else:
                                sentence        = str(img_display_path) + ","
                                sentence        = sentence + "No embedding classifier was detected!,"
                                identity        = ""
                                probabilities   = ""
                                spent           = str(time.time() - start_time)
                                spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                                sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                                one_image.append(sentence)
                    else:
                        sentence        = str(img_display_path) + ","
                        sentence        = sentence + "No human face was detected!,"
                        identity        = ""
                        probabilities   = ""
                        spent           = str(time.time() - start_time)
                        spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                        sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                        one_image.append(sentence)
                else:
                    sentence        = str(img_display_path) + ","
                    sentence        = sentence + "No human face was detected!,"
                    identity        = ""
                    probabilities   = ""
                    spent           = str(time.time() - start_time)
                    spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                    sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                    one_image.append(sentence)
            else:
                sentence        = str(img_display_path) + ","
                sentence        = sentence + "This file is not secure file! Please upload image file with extension png、 jpg jpeg!,"
                identity        = ""
                probabilities   = ""
                spent           = str(time.time() - start_time)
                spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                one_image.append(sentence) 
            display.append(one_image)
        return render_template(
            template_name_or_list='predict_batch_result.html',
            display=display
        )      
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="POST HTTP method required!"
        )

@app.route("/findSimilarKOLPage")
def find_similar_kol_page(): # manually upload single image file for prediction
    return render_template(template_name_or_list="find_similar_kol.html")

@app.route('/findSimilarKOLResult', methods=['POST', 'GET'])
def find_similar_kol_result(): # get multiple image prediction results | upload image files via POST request and feeds them to the FaceNet model to get prediction results
    renew(similar_image_savedir)
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="No 'file' field in POST request!"
            )
        file        = request.files['file']                                    # <FileStorage: 'download.jpg' ('image/jpeg')>
        filename    = secure_filename(''.join(lazy_pinyin(file.filename))) # download.jpg
        if filename == "":
            return render_template(
                template_name_or_list="warning.html",
                status="No selected file!"
            )
        upload_path = os.path.join(similar_image_savedir, filename)
        file.save(upload_path)
        image_paths = []
        image_paths.append(upload_path)
        img_display_path = os.path.join(similar_image_display_path, filename)
        kthKOL_similarity_timeused = []
        display   = kthKOL_similarity_timeused
        if file and allowed_img_file(filename=filename, allowed_img_set=allowed_img_set):
            tf.reset_default_graph()
            start_time  = time.time()
            images, count_per_image = load_and_align_data(image_paths) # count_per_image = [x] | x = 0时代表没有一个人脸被检测到            
            if count_per_image[0] != 0:
                if count_per_image[0] == "0":
                    return render_template(
                        template_name_or_list="warning.html",
                        status="The uploaded file is illegal. Please upload safe image file!"
                    )
                if count_per_image[0] != 1:
                    return render_template(
                        template_name_or_list="warning.html",
                        status="Please upload image which contains only one KOL face!"
                    )
                feed_dict               = {images_placeholder: images , phase_train_placeholder:False}
                emb                     = sess.run(embeddings, feed_dict=feed_dict)
                if images is not None: 
                    # with open(classifier_filename_exp, 'rb') as infile:
                    #     (model, class_names) = pickle.load(infile)
                    if model:
                        print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                        predictions                 = model.predict_proba(emb)
                        top_k_class_indices         = predictions[0].argsort()[-5:][::-1] # 最相似的前5个KOL
                        k = 0
                        for i in list(top_k_class_indices):
                            class_indices           = []
                            class_indices.append(i)
                            class_probabilities     = predictions[np.arange(len(class_indices)), class_indices] 
                            identity                = class_names[class_indices[0]]
                            probabilities           = class_probabilities[0]
                            probabilities           = str(probabilities).split('.')[0] + '.' + str(probabilities).split('.')[1][:3]
                            spent                   = str(time.time() - start_time)
                            spent                   = str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]
                            k                       += 1
                            kth_kol                 = str(img_display_path) + "," + str(k) + "th KOL: " + identity + "," + "Similarity: " + probabilities + "," + "Time consuming: " + spent + " seconds"
                            display.append(kth_kol)
                    else:
                        return render_template(
                            template_name_or_list="warning.html",
                            status="No embedding classifier was detected!"
                        )
                else:
                    return render_template(
                        template_name_or_list="warning.html",
                        status="No human face was detected!"
                    )
            else:
                return render_template(
                    template_name_or_list="warning.html",
                    status="No human face was detected!"
                )
        else:
            return render_template(
                template_name_or_list='warning.html',
                status="This file is not secure file! Please upload image file with extension png、 jpg、 jpeg!"
            ) 
        return render_template(
            template_name_or_list='find_similar_kol_result.html',
            display=display
        )      
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="POST HTTP method required!"
        )

if __name__ == '__main__':
    # tf.reset_default_graph()
    start_time  = time.time()
    with tf.Graph().as_default():
        graph = tf.get_default_graph()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 指定使用第一块GPU
        tf_config                                              =  tf.ConfigProto()
        tf_config.allow_soft_placement                         = True
        tf_config.gpu_options.per_process_gpu_memory_fraction  = 0.7
        tf_config.gpu_options.allow_growth                     = True
        with tf.Session(config=tf_config) as sess:
            image_size              = 160
            facenet.load_model(model_path)
            # pnet, rnet, onet = detect_face.create_mtcnn(sess, APP_ROOT + "\\src\\align")
            images_placeholder      = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings              = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            spend = time.time() - start_time
            print(str(spend)+' seconds | ' + str(spend/60) + ' minutes') # 30 seconds, 差不多需要花半分钟启动
            print("Server runing at locathost:%s!" %(port))
            serve(app=app, host="0.0.0.0", port=port)
            
            # global port 
            # port = 8888
            # print("Server runing at locathost:%s!" %(port))
            # serve(app=app, host='0.0.0.0', port=port)
            # serve(app=app, host='0.0.0.0', port=5000, debug=True, threaded=True)
            # app.run(host='0.0.0.0', debug=True, threaded=True)