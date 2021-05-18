import os
import cv2
import math
import time
import pickle
import shutil
import random
import pickle
import collections
import configparser
import numpy as np
import pandas as pd
from scipy import misc
import tensorflow as tf
from waitress import serve
import facenet.facenet as facenet
import align.detect_face as detect_face
from sklearn.svm import SVC
from flask import Flask, request, render_template, Response, redirect

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils import (
    get_dataset,
    renew,
    dir_tree,
    ImageClass,
    get_class_list,
    name_to_csv,
    xlsx_to_csv_pd,
    create,
    split_dataset,
    get_image_paths_and_labels
)

app                         = Flask(__name__)
app.secret_key              = os.urandom(24)
# APP_ROOT                  = os.path.dirname(os.path.abspath(__file__))

config                      = configparser.ConfigParser()
APP_ROOT                    = os.path.dirname(os.path.abspath(__file__))
config_file                 = os.path.join(APP_ROOT, "config.ini")
config.read(config_file)
project_root_folder         = config.get('Project', 'root_path')
KOL_dataset                 = config.get('KOL.dataset', 'path')
process_port                = config.getint('process.server', 'port')
# process_server_url        = "http://localhost:" + str(config.getint('process.server', 'port'))
model_path                  = os.path.join(project_root_folder, "models/20180402-114759/")
classifier_path             = os.path.join(project_root_folder, 'models/kol.pkl')
with open(classifier_path, 'rb') as infile:
    (model, class_names) = pickle.load(infile)
    print("Loaded classifier file")

@app.route('/') 
def index():
    """Video streaming home page."""
    process_obj = request.args.get('output_filename')
    back_url = "http://localhost:" + str(config.getint('main.server', 'port')) + '/trainModel'
    return render_template(template_name_or_list='index.html', param=process_obj, back_url=back_url)

@app.route("/processingProgress", methods=['POST', 'GET'])
def view_processing_progress(): 
    process_obj                 = request.args.get('param')
    align_clean_train_file      = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "align_clean_import_train.txt").replace('\\','/')
    align_clean_train_status    = open(align_clean_train_file, "r")
    lines                       = align_clean_train_status.readlines()
    align_status                = lines[0].split(',')[0]
    clean_status                = lines[0].split(',')[1]
    import_status               = lines[0].split(',')[2]
    train_status                = lines[0].split(',')[3]
    unzip_file                  = lines[0].split(',')[4]
    KOL_root_dir                = lines[0].split(',')[5]

    classn_imagen_align_clean_import_train = []
    status = ["Not started", "Start Processing", "Complete Processing", "Process Fail"] # 0 (创建), start (开始运行), end (完成), fail (失败)
    display = classn_imagen_align_clean_import_train
    dataset = dir_tree(os.path.join(unzip_file, KOL_root_dir).replace('\\','/'))
    if dataset == "0":
        return render_template(
            template_name_or_list="warning.html",
            status="Dataset doesn't exist. Please upload training zip file again!"
        )
    else:
        class_number = len(dataset)
        record = []
        record.append(str(class_number) + ',')
        one_class_image_number = []
        for i in range(len(dataset)):
            one_class_image_number.append(len(dataset[i]))
        image_number = sum(one_class_image_number)
        record.append(str(image_number) + ',')

        if align_status == '0':
            record.append(str(status[0]) + ',')
        if align_status == 'start':
            record.append(str(status[1]) + ',')
        if align_status == 'complete':
            record.append(str(status[2]) + ',')
        if align_status == 'fail':
            record.append(str(status[3]) + ',')
        if clean_status == '0':
            record.append(str(status[0]) + ',')
        if clean_status == 'start':
            record.append(str(status[1]) + ',')
        if clean_status == 'complete':
            record.append(str(status[2]) + ',')
        if clean_status == 'fail':
            record.append(str(status[3]) + ',')
        if import_status == '0':
            record.append(str(status[0]) + ',')
        if import_status == 'start':
            record.append(str(status[1]) + ',')
        if import_status == 'complete':
            record.append(str(status[2]) + ',')
        if import_status == 'fail':
            record.append(str(status[3]) + ',')
        if train_status == '0':
            record.append(str(status[0]) + ',')
        if train_status == 'start':
            record.append(str(status[1]) + ',')
        if train_status == 'complete':
            record.append(str(status[2]) + ',')
        if train_status == 'fail':
            record.append(str(status[3]) + ',')
        display = record[0] + record[1] + record[2] + record[3] + record[4] + record[5]
        return render_template(
            template_name_or_list='view_processing_progress.html', 
            param=process_obj,
            display=display,
            status=status
        )   

@app.route("/align")
def align(): 
    start_time                  = time.time()
    process_obj                 = request.args.get('param')
    align_dir                   = os.path.join(project_root_folder, "training_set/", process_obj, "align")
    renew(align_dir)
    align_clean_train_file      = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "align_clean_import_train.txt")
    align_clean_train_status    = open(align_clean_train_file, "r")
    lines                       = align_clean_train_status.readlines()
    clean_status                = lines[0].split(',')[1]
    import_status               = lines[0].split(',')[2]
    train_status                = lines[0].split(',')[3]
    unzip_file                  = lines[0].split(',')[4]
    KOL_root_dir                = lines[0].split(',')[5]

    dataset       = dir_tree(os.path.join(unzip_file, KOL_root_dir).replace('\\','/'))  
    if dataset == "0":
        return render_template(
            template_name_or_list="warning.html",
            status="Dataset doesn't exist. Please upload training zip file again!"
        )
    else:
        output_dir              = align_dir
        renew(output_dir)  

        with open(align_clean_train_file,'w') as f:    
            align_status = 'start'
            content = align_status + ',' + clean_status + ',' + import_status + ',' + train_status + ',' + unzip_file + ',' + KOL_root_dir
            f.write(content)

        random_order            = True
        random_key              = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        global graph
        with graph.as_default():
            # pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
            minsize     = 20 # minimum size of face
            threshold   = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
            factor      = 0.709 # scale factor
            margin      = 32
            with open(bounding_boxes_filename, "w") as text_file:
                nrof_images_total = 0
                nrof_successfully_aligned = 0
                multiple_face = 0
                if random_order:
                    random.shuffle(dataset)
                for cls in dataset:
                    output_class_dir = os.path.join(output_dir, cls.name)
                    if not os.path.exists(output_class_dir):
                        os.makedirs(output_class_dir)
                        if random_order:
                            random.shuffle(cls.image_paths)
                    count = 0
                    for image_path in cls.image_paths:
                        nrof_images_total += 1
                        filename = os.path.splitext(os.path.split(image_path)[1])[0]
                        output_filename = os.path.join(output_class_dir, filename+'.png')
                        print(nrof_images_total, image_path)
                        if not os.path.exists(output_filename):
                            try:
                                img = misc.imread(image_path)
                            except (IOError, ValueError, IndexError) as e:
                                errorMessage = '{}: {}'.format(image_path, e)
                                print(errorMessage)
                            else:
                                if img.ndim<2:
                                    print('Unable to align "%s"' % image_path)
                                    text_file.write('%s\n' % (output_filename))
                                    continue
                                if img.ndim == 2:
                                    img = facenet.to_rgb(img)
                                img = img[:,:,0:3]

                                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                                nrof_faces = bounding_boxes.shape[0]
                                if nrof_faces > 0:
                                    det = bounding_boxes[:,0:4]
                                    det_arr = []
                                    img_size = np.asarray(img.shape)[0:2]
                                    if nrof_faces>1:
                                        # 去看facenet源代码: 有--detect_multiple_faces的input选项，可以检测多张人脸并对齐提取出来(没有添加这个参数的话，就提取出一张人脸)
                                        # 我此处有修改: 为了方便green hand创建训练集减少人工操作，如果非1张人脸就不提取，图片中只有一张人脸才提取
                                        multiple_face += 1
                                        print("{} pictures contain multiple faces, and this picture contains {} faces".format(multiple_face, nrof_faces))
                                    else:
                                        count += 1
                                        det_arr.append(np.squeeze(det))
                                        for i, det in enumerate(det_arr):
                                            det = np.squeeze(det)
                                            bb = np.zeros(4, dtype=np.int32)
                                            bb[0] = np.maximum(det[0]-margin/2, 0)
                                            bb[1] = np.maximum(det[1]-margin/2, 0)
                                            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                                            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                                            nrof_successfully_aligned += 1
                                            filename_base, file_extension = os.path.splitext(output_filename)
                                            # output_filename_n = "{}{}".format(filename_base, file_extension)
                                            align_image_name = "{}{}".format(cls.name + "_" + str(count).zfill(5), file_extension)
                                            output_filename_n = os.path.join(output_dir, cls.name, align_image_name)
                                            misc.imsave(output_filename_n, scaled)
                                            text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                                else:
                                    print('Unable to align "%s"' % image_path)
                                    text_file.write('%s\n' % (output_filename))
                                
        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
        class_number = len(dataset)
        spend = "%.3f seconds = %.3f minutes = %.3f hrs" %((time.time() - start_time), (time.time() - start_time) / 60, (time.time() - start_time) / 3600)
        process_efficiency = "%.3f seconds/image" %((time.time() - start_time) / nrof_images_total)
        display = []
        display.append(class_number)
        display.append(nrof_images_total)
        display.append(nrof_successfully_aligned)
        display.append(spend)
        display.append(process_efficiency)
        with open(align_clean_train_file,'w') as f:    
            align_status = 'complete'
            content = align_status + ',' + clean_status + ',' + import_status + ',' + train_status + ',' + unzip_file + ',' + KOL_root_dir
            f.write(content)
        print("================Complete Align Action================")
        return render_template(
            template_name_or_list='align_result.html',
            param=process_obj,
            display=display
        )  
            
@app.route("/clean")
def clean(): 
    start_time                  = time.time()
    process_obj                 = request.args.get('param')
    align_dir                   = os.path.join(project_root_folder, "training_set/", process_obj, "align")
    clean_dir                   = os.path.join(project_root_folder, "training_set/", process_obj, "clean")
    renew(clean_dir)
    sentence                    = []
    align_clean_train_file      = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "align_clean_import_train.txt")
    align_clean_train_status    = open(align_clean_train_file, "r")
    lines                       = align_clean_train_status.readlines()
    align_status                = lines[0].split(',')[0]
    import_status               = lines[0].split(',')[2]
    train_status                = lines[0].split(',')[3]
    unzip_file                  = lines[0].split(',')[4]
    KOL_root_dir                = lines[0].split(',')[5]
    output_dir                  = clean_dir
    renew(output_dir)

    with open(align_clean_train_file,'w') as f:    
        clean_status = 'start'
        content = align_status + ',' + clean_status + ',' + import_status + ',' + train_status + ',' + unzip_file + ',' + KOL_root_dir
        f.write(content)

    global graph
    with graph.as_default():
        minsize             = 20
        threshold           = [0.6, 0.7, 0.7]
        factor              = 0.709

        in_dir = os.path.join(align_dir)
        classes = [path for path in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, path))]
        classes.sort()
        nrof_classes        = len(classes)
        cleanset            = []
        nrof_images_total   = 0
        for i in range(nrof_classes):
            class_name      = classes[i]
            facedir         = os.path.join(in_dir, class_name)
            image_paths     = []
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths     = [os.path.join(facedir, img) for img in images]
                class_imageProb = dict()
                clean_paths     = []                
                Top_KOL_name    = []
                person_detected = collections.Counter()
                iterate_paths = []
                for image_path in image_paths:
                    nrof_images_total += 1
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]
                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        if bounding_boxes == '0':
                            print("Note! Do not detect one face!!!!!!!!")
                            continue                        
                        else:
                            nrof_faces = bounding_boxes.shape[0]
                            # if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            bb = np.zeros((nrof_faces, 4), dtype=np.int32)
                            for i in range(nrof_faces):
                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]
                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(img[0]) or bb[i][3] >= len(img):
                                    print('face is inner of range!')
                                    continue
                                else:
                                    iterate_paths.append(image_path)
                                    cropped = img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                    scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
                                    scaled = facenet.prewhiten(scaled)

                                    scaled_reshape = scaled.reshape(-1, image_size, image_size, 3)
                                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                                    predictions = model.predict_proba(emb_array)
                                    best_class_indices = np.argmax(predictions, axis=1)

                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                    best_name = class_names[best_class_indices[0]]
                                    # print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                                    class_imageProb.setdefault(best_name, []).append(best_class_probabilities)
                                    print(class_name, image_paths.index(image_path), best_name, best_class_probabilities)

                                    if best_class_probabilities > 0.09:
                                        person_detected[best_name] += 1
                base_image = []
                # if len(person_detected.most_common(1)) == 0 and len(iterate_paths) != 0:
                #     base_image.append(iterate_paths[0])
                # if len(person_detected.most_common(1)) == 0 and len(iterate_paths) == 0: 
                #     base_image.append(image_paths[0])
                # else:
                #     for name, count in person_detected.most_common(1):
                #         Top_KOL_name.append(name)
                #         Top_KOL = Top_KOL_name[0]
                #         Top_KOL_max_prob = class_imageProb[Top_KOL]
                #         Top_KOL_max_prob_index = Top_KOL_max_prob.index(max(Top_KOL_max_prob))
                #         Top_KOL_max_prob_image_path = image_paths[Top_KOL_max_prob_index]
                #         base_image.append(Top_KOL_max_prob_image_path)
                #         # print("base image 是: ", Top_KOL_max_prob_image_path)
                base_image.append(image_paths[0])
                image_files = ['top_kol', 'current_image']
                image_files[0] = base_image[0]
                for image_path in iterate_paths:
                    image_files[1] = image_path
                    img_list = []
                    margin   = 44
                    for image in image_files:
                        img         = misc.imread(os.path.expanduser(image), mode='RGB')
                        img_size    = np.asarray(img.shape)[0:2]
                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        if len(bounding_boxes) < 1:
                            image_paths.remove(image)
                            print("can't detect face, remove ", image)
                            continue
                        else:
                            det = np.squeeze(bounding_boxes[0, 0:4])
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - margin / 2, 0)
                            bb[1] = np.maximum(det[1] - margin / 2, 0)
                            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                            prewhitened = facenet.prewhiten(aligned)
                            img_list.append(prewhitened)
                    images = np.stack(img_list)
                    feed_dict   = {images_placeholder: images, phase_train_placeholder:False }
                    emb         = sess.run(embeddings, feed_dict=feed_dict)
                    dist        = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :]))))
                    print("class: " + class_name + ", index: " + str(image_paths.index(image_path)).zfill(5) + ", distance: " + str(dist))
                    if dist < 1.06:
                        clean_paths.append(image_path)
                cleanset.append(ImageClass(class_name, clean_paths))
        classn_imagen_cleann_time_efficiency = []
        classn_imagen_cleann_time_efficiency.append(len(cleanset))
        classn_imagen_cleann_time_efficiency.append(nrof_images_total)
        one_class_image_number = []
        for i in range(len(cleanset)):
            one_class_image_number.append(len(cleanset[i]))
        clean_image_number = sum(one_class_image_number)
        classn_imagen_cleann_time_efficiency.append(clean_image_number)
        sentence.append(classn_imagen_cleann_time_efficiency)
        spend = "%.3f seconds = %.3f minutes = %.3f hrs" %((time.time() - start_time), (time.time() - start_time) / 60, (time.time() - start_time) / 3600)
        classn_imagen_cleann_time_efficiency.append(spend)
        process_efficiency = "%.3f seconds/image" %((time.time() - start_time) / nrof_images_total)
        classn_imagen_cleann_time_efficiency.append(process_efficiency)

        for cls in cleanset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)  
            count = 0
            for image_path in cls.image_paths:
                count += 1
                filename_base, file_extension = os.path.splitext(image_path)
                clean_image_name = "{}{}".format(cls.name + "_" + str(count).zfill(5), file_extension) 
                clean_image_path = os.path.join(output_class_dir, clean_image_name)
                print("Save clean face: ", str(count).zfill(5), clean_image_path)
                shutil.copy(image_path, clean_image_path)   # 复制文件及权限, Copy data and mode bits

    display = sentence[0]
    with open(align_clean_train_file,'w') as f:    
        clean_status = 'complete'
        content = align_status + ',' + clean_status + ',' + import_status + ',' + train_status + ',' + unzip_file + ',' + KOL_root_dir
        f.write(content)
    print("================Complete Clean Action================")
    return render_template(
        template_name_or_list='clean_result.html',
        param=process_obj,
        display=display
    )  

@app.route("/import_clean")
def import_clean(): 
    start_time                  = time.time()
    process_obj                 = request.args.get('param')
    clean_dir                   = os.path.join(project_root_folder, "training_set/", process_obj, "clean")
    import_dir                  = KOL_dataset
    sentence                    = []
    align_clean_train_file      = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "align_clean_import_train.txt")
    align_clean_train_status    = open(align_clean_train_file, "r")
    lines                       = align_clean_train_status.readlines()
    align_status                = lines[0].split(',')[0]
    clean_status                = lines[0].split(',')[1]
    train_status                = lines[0].split(',')[3]
    unzip_file                  = lines[0].split(',')[4]
    KOL_root_dir                = lines[0].split(',')[5]
    output_dir                  = KOL_dataset

    with open(align_clean_train_file,'w') as f:    
        import_status = 'start'
        content = align_status + ',' + clean_status + ',' + import_status + ',' + train_status + ',' + unzip_file + ',' + KOL_root_dir
        f.write(content)

    if not os.path.exists(output_dir):
        return render_template(
            template_name_or_list="warning.html",
            status="Training set pool path doesn't exist, please check and config with correct path!"
        )
    else:
        xlsx_file           = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "KOL_UUID.xlsx")
        xlsx_to_csv_pd(xlsx_file)
        csv_file            = name_to_csv(xlsx_file)
        if not os.path.exists(csv_file):
            return render_template(
                template_name_or_list="warning.html",
                status="UUID-Name mapping file doesn't exist, please check and upload with it!"
            )
        else:
            dataset_clean       = dir_tree(os.path.join(clean_dir).replace('\\','/'))
            dataset_pool        = dir_tree(output_dir)
            class_name_clean    = get_class_list(dataset_clean)
            class_name_pool     = get_class_list(dataset_pool)
            same_class_uuid     = [x for x in class_name_clean if x in class_name_pool]
            add_class_uuid      = [x for x in class_name_clean if x not in class_name_pool]

            same_uuid_name      = []
            add_uuid_name       = []
            df                  = pd.read_csv(csv_file, encoding="utf-8")
            for uuid in same_class_uuid:
                row_index = df[df["UUID"]==uuid].index.tolist()[0]
                name = df["Name"].iloc[row_index]
                same_uuid_name.append([uuid, name])
            for uuid in add_uuid_name:
                row_index = df[df["UUID"]==uuid].index.tolist()[0]
                name = df["Name"].iloc[row_index]
                add_uuid_name.append([uuid, name])
            print("================Give user selection to choose which kol to train================")
            return render_template(
                template_name_or_list="import_clean.html",
                param=process_obj,
                same_uuid_name=same_uuid_name,
                same_identity=len(same_uuid_name)
            )

@app.route("/import_result", methods=['POST','GET'])
def import_result(): 
    start_time                  = time.time()
    process_obj                 = request.args.get('output_filename')
    clean_dir                   = os.path.join(project_root_folder, "training_set/", process_obj, "clean")
    arrange_dir                 = os.path.join(project_root_folder, "training_set/", process_obj, "arrange")
    create(arrange_dir)
    import_dir                  = KOL_dataset
    align_clean_train_file      = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "align_clean_import_train.txt")
    align_clean_train_status    = open(align_clean_train_file, "r")
    lines                       = align_clean_train_status.readlines()
    align_status                = lines[0].split(',')[0]
    clean_status                = lines[0].split(',')[1]
    train_status                = lines[0].split(',')[3]
    unzip_file                  = lines[0].split(',')[4]
    KOL_root_dir                = lines[0].split(',')[5]
    output_dir                  = KOL_dataset

    dataset_clean       = dir_tree(os.path.join(clean_dir).replace('\\','/'))
    dataset_pool        = dir_tree(output_dir)
    class_name_clean    = get_class_list(dataset_clean)
    class_name_pool     = get_class_list(dataset_pool)
    add_class_uuid      = [x for x in class_name_clean if x not in class_name_pool]
    same_class_uuid     = [x for x in class_name_clean if x in class_name_pool] 

    dataset             = dir_tree(output_dir)
    class_number        = len(dataset)
    one_class_image_number = []
    for i in range(len(dataset)):
        one_class_image_number.append(len(dataset[i]))
    image_number        = sum(one_class_image_number)
    before_import       = [class_number, image_number]

    if request.method == "POST":
        select_option               = request.values.getlist("select")
        data                        = request.values.getlist("s_option")
        select_content              = []
        if select_option[0] == "All":
            select_content.append(same_class_uuid)
        elif select_option[0] == "None":
            select_content.append([])
        else:
            select_content.append(data)
        user_select_exist_uuid  = select_content[0]
        for uuid in add_class_uuid:
            original_path       = os.path.join(clean_dir, uuid)
            dest_path           = os.path.join(output_dir, uuid)
            shutil.copytree(original_path, dest_path)
        for uuid in user_select_exist_uuid:
            clean_uuid_dir      = os.path.join(clean_dir, uuid).replace('\\','/')
            arrang_uuid_dir     = os.path.join(arrange_dir, uuid).replace('\\','/')
            pool_uuid_dir       = os.path.join(output_dir, uuid).replace('\\','/')
            create(arrang_uuid_dir)
            for image in os.listdir(pool_uuid_dir):
                image_path      = os.path.join(pool_uuid_dir, image).replace('\\','/')
                shutil.move(image_path, arrang_uuid_dir)
            if not os.path.exists(pool_uuid_dir):
                os.mkdir(pool_uuid_dir)
            arrange_file        = []
            for image in os.listdir(clean_uuid_dir):
                image_path      = os.path.join(clean_uuid_dir, image).replace('\\','/')
                arrange_file.append(image_path)
            for image in os.listdir(arrang_uuid_dir):
                image_path      = os.path.join(arrang_uuid_dir, image).replace('\\','/')
                arrange_file.append(image_path)
            count               = 0
            for path in arrange_file:
                count           += 1
                _, file_extension = os.path.splitext(path)
                pool_image_name = "{}{}".format(uuid + "_" + str(count).zfill(5), file_extension)
                pool_uuid_path  = os.path.join(pool_uuid_dir, pool_image_name).replace('\\','/')
                shutil.copy(path, pool_uuid_path)    

        xlsx_file           = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "KOL_UUID.xlsx")
        csv_file            = name_to_csv(xlsx_file)
        df                  = pd.read_csv(csv_file, encoding="utf-8")
        add_class_name      = []
        for uuid in add_class_uuid:
            row_index       = df[df["UUID"]==uuid].index.tolist()[0]
            name            = df["Name"].iloc[row_index]
            add_class_name.append(name)
        select_exist_name   = []
        for uuid in user_select_exist_uuid:
            row_index       = df[df["UUID"]==uuid].index.tolist()[0]
            name            = df["Name"].iloc[row_index]
            select_exist_name.append(name)
        name_list           = add_class_name + select_exist_name
        import_class_number = len(name_list) 

        import_image_number = 0
        for uuid in (add_class_uuid + user_select_exist_uuid):
            uuid_dir        = os.path.join(clean_dir, uuid)
            for path in os.listdir(uuid_dir):
                import_image_number += 1

        import_summary      = [add_class_name, select_exist_name, import_class_number, import_image_number]

        dataset             = dir_tree(output_dir)
        class_number        = len(dataset)
        one_class_image_number = []
        for i in range(len(dataset)):
            one_class_image_number.append(len(dataset[i]))
        image_number        = sum(one_class_image_number)
        after_import        = [class_number, image_number]

        with open(align_clean_train_file,'w') as f:    
            import_status = 'complete'
            content = align_status + ',' + clean_status + ',' + import_status + ',' + train_status + ',' + unzip_file + ',' + KOL_root_dir
            f.write(content)
        print("================Complete Dataset Import================")
    
        # zip_obj             = os.path.join(project_root_folder, "training_set/", process_obj)
        # if os.path.exists(zip_obj):
        #     shutil.rmtree(zip_obj)

        return render_template(
            template_name_or_list="import_result.html",
            param=process_obj,
            before_import=before_import,
            import_summary=import_summary,
            after_import=after_import
        )
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="POST HTTP method required!"
        )

@app.route("/train_model")
def train_model(): 
    start_time                  = time.time()
    process_obj                 = request.args.get('param')
    data_dir                    = KOL_dataset
    align_clean_train_file      = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "align_clean_import_train.txt")
    align_clean_train_status    = open(align_clean_train_file, "r")
    lines                       = align_clean_train_status.readlines()
    align_status                = lines[0].split(',')[0]
    clean_status                = lines[0].split(',')[1]
    import_status               = lines[0].split(',')[2]
    unzip_file                  = lines[0].split(',')[4]
    KOL_root_dir                = lines[0].split(',')[5]

    with open(align_clean_train_file,'w') as f:    
        train_status = 'start'
        content = align_status + ',' + clean_status + ',' + import_status + ',' + train_status + ',' + unzip_file + ',' + KOL_root_dir
        f.write(content)

    np.random.seed(seed=666)
    dataset_tmp             = get_dataset(data_dir)
    train_set, test_set     = split_dataset(dataset_tmp)
    dataset                 = train_set
    unqualified_kol         = []
    for cls in dataset:
        if len(cls.image_paths) <= 0:
            unqualified_kol.append(cls.name)
        # assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
    back_url = "http://localhost:" + str(config.getint('main.server', 'port')) + '/trainModel'
    if len(unqualified_kol) != 0:
        return render_template(
            template_name_or_list='train_warning1.html',
            param=process_obj,
            unqualified_kol=unqualified_kol,
            back_url=back_url
        )    
    else:
        paths, labels   = get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))

        print('Loading feature extraction model')
        embedding_size  = embeddings.get_shape()[1]

        print('Calculating features for images')
        nrof_images     = len(paths)
        # Number of images to process in a batch
        batch_size      = 90
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array       = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i*batch_size
            end_index   = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images      = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict   = { images_placeholder:images, phase_train_placeholder:False }
            emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
        classifier_filename     = os.path.join(config.get('KOL.model', 'new_path', raw = 0), "KOL.pkl")
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        csv_file        = os.path.join(project_root_folder, "training_set/", process_obj, "txt", "KOL_UUID.csv")
        df              = pd.read_csv(csv_file, encoding="utf-8")

        print('Training classifier')
        model           = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        class_names     = []
        for cls in dataset:
            uuid        = cls.name
            row_index   = df[df["UUID"]==uuid].index.tolist()[0]
            name        = df["Name"].iloc[row_index]
            class_names.append(name)

        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

        with open(align_clean_train_file,'w') as f:    
            train_status = 'complete'
            content = align_status + ',' + clean_status + ',' + import_status + ',' + train_status + ',' + unzip_file + ',' + KOL_root_dir
            f.write(content)

        return render_template(
            template_name_or_list="train_result.html",
            param=process_obj
        )


if __name__ == '__main__':
    start_time  = time.time()
    with tf.Graph().as_default():
        graph = tf.get_default_graph()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 指定使用第一块GPU
        tf_config                                              =  tf.ConfigProto()
        tf_config.allow_soft_placement                         = True
        tf_config.gpu_options.per_process_gpu_memory_fraction  = 0.7
        tf_config.gpu_options.allow_growth                     = True
        with tf.Session(config=tf_config) as sess:
            image_size              = 160
            facenet.load_model(model_path)
            pnet, rnet, onet        = detect_face.create_mtcnn(sess, None)
            images_placeholder      = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings              = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            spend = time.time() - start_time
            print(str(spend)+' seconds | ' + str(spend/60) + ' minutes')
            print("Server runing at locathost:%s!" %(process_port))
            serve(app=app, host="0.0.0.0", port=process_port)
            
            # app.run(host='0.0.0.0', debug=True, threaded=True)