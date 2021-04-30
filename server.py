import os
import cv2
import time
import pickle
import shutil
import numpy as np
import tensorflow as tf
from waitress import serve
from scipy.misc import imread
import facenet.facenet as facenet
import align.detect_face as detect_face
from pypinyin import lazy_pinyin
from werkzeug.utils import secure_filename
from imutils.video import WebcamVideoStream
from flask import Flask, request, render_template

from utils import (
    allowed_set,
    allowed_file,
    load_and_align_data
)

tf.reset_default_graph()

app             = Flask(__name__)
app.secret_key  = os.urandom(24)
APP_ROOT        = os.path.dirname(os.path.abspath(__file__))
uploads_path    = os.path.join(APP_ROOT, 'static')

@app.route("/")
def index_page():   # select single predict mode or batch predict mode
    return render_template(template_name_or_list="index.html")

@app.route("/predictSinglePage")
def predict_single_page(): # manually upload single image file for prediction
    return render_template(template_name_or_list="predict_single.html")

@app.route('/predictSingleImage', methods=['POST', 'GET'])
def predict_single_image(): # get multiple image prediction results | upload image files via POST request and feeds them to the FaceNet model to get prediction results
    images_savedir = "static/"
    if  os.path.exists(images_savedir):
        shutil.rmtree(images_savedir)
    if not os.path.exists(images_savedir):
        os.makedirs(images_savedir)
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
        upload_path = os.path.join(uploads_path, filename)
        file.save(upload_path)
        static_path             = "static/" + filename
        image_paths = []
        image_paths.append(static_path)
        path_nofperson_identity_similarity_timeused = []
        display = path_nofperson_identity_similarity_timeused
        if file and allowed_file(filename=filename, allowed_set=allowed_set):
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
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                if images is not None: 
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)
                    if model:
                        print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                        predictions                 = model.predict_proba(emb)
                        best_class_indices          = np.argmax(predictions, axis=1) # <class 'numpy.ndarray'> [0]
                        best_class_probabilities    = predictions[np.arange(len(best_class_indices)), best_class_indices] # [0.99692782]
                        k = 0
                        for j in range(count_per_image[0]):
                            sentence        = str(static_path) + ","
                            sentence        = sentence + str(count_per_image[0]) + " people detected!,"
                            print("\npeople in image %s :" %(filename), '%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                            identity        = class_names[best_class_indices[k]]
                            probabilities   = best_class_probabilities[k]
                            k+=1
                            probabilities   = "Similarity: " + str(probabilities).split('.')[0] + '.' + str(probabilities).split('.')[1][:3]
                            spent           = str(time.time() - start_time)
                            spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                            sentence        = sentence + "Person " + str(k) + ": " + str(identity) + "," + str(probabilities) + "," + str(spent)
                            display.append(sentence)
                    else:
                        sentence        = str(static_path) + ","
                        sentence        = sentence + "No embedding classifier was detected!,"
                        identity        = ""
                        probabilities   = ""
                        spent           = str(time.time() - start_time)
                        spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                        sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                        display.append(sentence)
                else:
                    sentence        = str(static_path) + ","
                    sentence        = sentence + "No human face was detected!,"
                    identity        = ""
                    probabilities   = ""
                    spent           = str(time.time() - start_time)
                    spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                    sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                    display.append(sentence)
            else:
                sentence        = str(static_path) + ","
                sentence        = sentence + "No human face was detected!,"
                identity        = ""
                probabilities   = ""
                spent           = str(time.time() - start_time)
                spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                display.append(sentence)
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
    images_savedir = "static/"
    if  os.path.exists(images_savedir):
        shutil.rmtree(images_savedir)
    if not os.path.exists(images_savedir):
        os.makedirs(images_savedir)
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
            if allowed_file(filename=filename, allowed_set=allowed_set):
                if filename == "":
                    return render_template(
                        template_name_or_list="warning.html",
                        status="No selected file!"
                    )
                upload_path             = os.path.join(uploads_path, filename)
                file.save(upload_path)
                static_path             = "static/" + filename
                image_paths = []
                image_paths.append(static_path)
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
                    classifier_filename_exp = os.path.expanduser(classifier_filename)
                    if images is not None: 
                        with open(classifier_filename_exp, 'rb') as infile:
                            (model, class_names) = pickle.load(infile)
                            if model:
                                print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
                                predictions                 = model.predict_proba(emb)
                                best_class_indices          = np.argmax(predictions, axis=1)
                                best_class_probabilities    = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                k = 0
                                for j in range(count_per_image[0]):
                                    sentence        = str(static_path) + ","
                                    sentence        = sentence + str(count_per_image[0]) + " people detected!,"
                                    print("\npeople in image %s :" %(filename), '%s: %.3f' % (class_names[best_class_indices[k]], best_class_probabilities[k]))
                                    identity        = class_names[best_class_indices[k]]
                                    probabilities   = best_class_probabilities[k]
                                    k+=1
                                    probabilities   = "Similarity: " + str(probabilities).split('.')[0] + '.' + str(probabilities).split('.')[1][:3]
                                    spent           = str(time.time() - start_time)
                                    spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                                    sentence        = sentence + "Person " + str(k) + ": " + str(identity) + "," + str(probabilities) + "," + str(spent)
                                    one_image.append(sentence)
                            else:
                                sentence        = str(static_path) + ","
                                sentence        = sentence + "No embedding classifier was detected!,"
                                identity        = ""
                                probabilities   = ""
                                spent           = str(time.time() - start_time)
                                spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                                sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                                one_image.append(sentence)
                    else:
                        sentence        = str(static_path) + ","
                        sentence        = sentence + "No human face was detected!,"
                        identity        = ""
                        probabilities   = ""
                        spent           = str(time.time() - start_time)
                        spent           = "Time consuming: " + str(spent).split('.')[0] + '.' + str(spent).split('.')[1][:2]  + " seconds"
                        sentence        = sentence + str(identity) + "," + str(probabilities) + "," + str(spent)
                        one_image.append(sentence)
                else:
                    sentence        = str(static_path) + ","
                    sentence        = sentence + "No human face was detected!,"
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
    images_savedir = "static/"
    if  os.path.exists(images_savedir):
        shutil.rmtree(images_savedir)
    if not os.path.exists(images_savedir):
        os.makedirs(images_savedir)
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
        upload_path = os.path.join(uploads_path, filename)
        file.save(upload_path)
        static_path             = "static/" + filename
        image_paths = []
        image_paths.append(static_path)
        kthKOL_similarity_timeused = []
        display   = kthKOL_similarity_timeused
        if file and allowed_file(filename=filename, allowed_set=allowed_set):
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
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                if images is not None: 
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)
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
                            kth_kol                 = str(static_path) + "," + str(k) + "th KOL: " + identity + "," + "Similarity: " + probabilities + "," + "Time consuming: " + spent + " seconds"
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
    with tf.Graph().as_default():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 指定使用第一块GPU
        config                                              =  tf.ConfigProto()
        config.allow_soft_placement                         = True
        config.gpu_options.per_process_gpu_memory_fraction  = 0.7
        config.gpu_options.allow_growth                     = True
        with tf.Session(config=config) as sess:
            model               = "models/20180402-114759/"
            classifier_filename = "models/kol.pkl"
            image_size          = 160
            facenet.load_model(model)
            images_placeholder      = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings              = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            serve(app=app, host='0.0.0.0', port=5000)