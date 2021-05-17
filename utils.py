import os
import cv2
import shutil
import zipfile
import numpy as np
from scipy import misc
import tensorflow as tf
import align.detect_face
from six.moves import xrange
import facenet.facenet as facenet
from scipy.misc import imresize, imsave
from tensorflow.python.platform import gfile
from align.detect_face import detect_face
from align.detect_face import create_mtcnn

allowed_img_set = set(['png', 'jpg', 'jpeg'])           # allowed image formats for upload
allowed_video_set = set(['flv', 'avi', 'mp4', 'mov', 'wmv'])

def allowed_img_file(filename, allowed_img_set):
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_img_set
    return check

def allowed_video_file(filename, allowed_video_set):
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_video_set
    return check


def get_face(img, pnet, rnet, onet, image_size):
    minsize             = 20
    threshold           = [0.6, 0.7, 0.7]
    factor              = 0.709
    margin              = 44
    input_image_size    = image_size
    img_size            = np.asarray(img.shape)[0:2]
    bounding_boxes, _   = detect_face(
        img=img, minsize=minsize, pnet=pnet, rnet=rnet,
        onet=onet, threshold=threshold, factor=factor
    )

    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            det         = np.squeeze(face[0:4])
            bb          = np.zeros(4, dtype=np.int32)
            bb[0]       = np.maximum(det[0] - margin / 2, 0)
            bb[1]       = np.maximum(det[1] - margin / 2, 0)
            bb[2]       = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3]       = np.minimum(det[3] + margin / 2, img_size[0])
            cropped     = img[bb[1]: bb[3], bb[0]:bb[2], :]
            face_img    = imresize(arr=cropped, size=(input_image_size, input_image_size), mode='RGB')
            return face_img
    else:
        return None


def load_image(img, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    image = np.zeros((1, image_size, image_size, 3))
    if img.ndim == 2:
            img = to_rgb(img)
    if do_prewhiten:
            img = prewhiten(img)
    img = crop(img, do_random_crop, image_size)
    img = flip(img, do_random_flip)
    image[:, :, :, :] = img
    return image


def forward_pass(img, session, images_placeholder, phase_train_placeholder, embeddings, image_size):
    if img is not None:
        image = load_image(
            img=img, do_random_crop=False, do_random_flip=False,
            do_prewhiten=True, image_size=image_size
        )
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        embedding = session.run(embeddings, feed_dict=feed_dict)
        return embedding
    else:
        return None


def load_embeddings():
    embedding_dict = defaultdict()  
    for embedding in glob.iglob(pathname='embeddings/*.npy'):  
        name                    = remove_file_extension(embedding)
        dict_embedding          = np.load(embedding)
        embedding_dict[name]    = dict_embedding
    return embedding_dict


def remove_file_extension(filename):
    filename = os.path.splitext(filename)[0]
    return filename


def identify_face(embedding, embedding_dict):
    min_distance = 100
    try:
        for (name, dict_embedding) in embedding_dict.items():
            distance = np.linalg.norm(embedding - dict_embedding)
            if distance < min_distance:
                min_distance = distance
                identity = name
        if min_distance <= 1.1:
            identity = identity[11:]
            result = str(identity) + " with distance: " + str(min_distance)
            return result
        else:
            result = "Not in the database, the distance is " + str(min_distance)
            return result
    except Exception as e:
        print(str(e))
        return str(e)


def load_model(model, input_map=None):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def load_and_align_data(image_paths):

    minsize     = 20 
    threshold   = [ 0.6, 0.7, 0.7 ]  
    factor      = 0.709 
    image_size  = 160
    margin      = 44
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        config                                              =  tf.ConfigProto()
        config.allow_soft_placement                         = True
        config.gpu_options.per_process_gpu_memory_fraction  = 0.7
        config.gpu_options.allow_growth                     = True
        sess                                                = tf.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [] 
    count_per_image = []
    for i in xrange(nrof_samples):
        # img = misc.imread(name=file, mode='RGB')
        img = misc.imread(os.path.expanduser(image_paths[i])) # (157, 320, 3)
        if img.shape[2] !=3:
            print("Cannot feed value of shape (x, y, z, 4) for Tensor 'pnet/input:0', which has shape '(?, ?, ?, 3)'")
            images = "0"
            count_per_image.append("0")
            return images, count_per_image
        img_size = np.asarray(img.shape)[0:2] 
        bounding_boxes, _ = detect_face(img, minsize, pnet, rnet, onet, threshold, factor) # (3, 5) | 3代表检测到3个人脸
        count_per_image.append(len(bounding_boxes)) # bounding_boxes.shape = (x, 5) | x = 0时代表没有一个人脸被检测到
        if len(bounding_boxes) == 0:
            print("No person detected in this image!")
            images = "0"
            return images, count_per_image
        else:
            for j in range(len(bounding_boxes)):	
                det = np.squeeze(bounding_boxes[j,0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned) # (160, 160, 3)
                img_list.append(prewhitened)	
            images = np.stack(img_list) # (3, 160, 160, 3)
            return images, count_per_image

def format_dataset(file_name):
    zip_file = zipfile.ZipFile(file_name)
    format_zip = []
    for names in zip_file.namelist():
        if os.path.splitext(names)[-1] == '.zip':
            format_zip.append(0)
        else:
            format_zip.append(1)
    return format_zip
    
def un_zip(file_name):
    """unzip zip file"""
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(os.path.splitext(file_name)[0]):
        pass
    else:
        os.mkdir(os.path.splitext(file_name)[0])
    for names in zip_file.namelist():
        zip_file.extract(names, os.path.splitext(file_name)[0])
    zip_file.close()
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        print("The file " + file_name + " does not exist")

def create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def renew(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

