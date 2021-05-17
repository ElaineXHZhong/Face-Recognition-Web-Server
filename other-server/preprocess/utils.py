import os
import shutil
import pandas as pd

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp.replace('\\','/')) \
                    if os.path.isdir(os.path.join(path_exp, path).replace('\\','/'))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def renew(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

def dir_tree(path): # C:\Kooler\training_set/upload/automation
    if not os.path.exists(path):
        print("dataset path doesn't exist!")
        return "0"
    else:
        dataset = get_dataset(path)
        return dataset

def get_class_list(dataset):
    class_list = []
    for cls in dataset: 
        class_list.append(cls.name)
    return class_list

def name_to_csv(xlsx_file):
    (path, filename)    = os.path.split(xlsx_file)
    (file, ext)         = os.path.splitext(filename)
    csv_file            = os.path.join(path, file + ".csv").replace('\\','/')
    return csv_file

def xlsx_to_csv_pd(xlsx_file):
    data_xls            = pd.read_excel(xlsx_file.replace('\\','/'), index_col=0)
    csv_file            = name_to_csv(xlsx_file.replace('\\','/'))
    data_xls.to_csv(csv_file, encoding='utf-8')