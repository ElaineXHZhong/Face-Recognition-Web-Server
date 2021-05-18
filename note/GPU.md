# GPU Environment Configuration

以下两种环境均可:
- `Visual Studio 2017 + python 3.6.12 + tensorflow-gpu 1.7.0 + CUDA 9.0 + cuDNN 7.0.5 + facenet site-packages`
- `CUDA 11.0 + tensorflow-gpu 1.14.0`
- `CUDA 11.0 + tensorflow-gpu 2.4.0`
    ```markdown
    - 修改文件:
        1. facenet/src/align/align_dataset_mtcnn.py
        2. anaconda3/envs/facenet/lib/python3.6/site-packages/align/detect_face.py
    - 修改内容: 把所有的 tf. 替换为 tf.compat.v1.  (兼容性处理: tf.compat允许您编写在TensorFlow 1.x和2.x中均可使用的代码)

    - 修改文件: anaconda3/envs/facenet/lib/python3.6/site-packages/align/detect_face.py
    - 修改内容: 把第194行的 feed_in, dim = (inp, input_shape[-1].value) 替换为 feed_in, dim = (inp, input_shape[-1])
    ```

## Prerequisite software

配置前先安装一些必备软件:
- [Git](https://git-scm.com/downloads)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Anaconda](https://www.anaconda.com/products/individual#Downloads)
- [7 Zip](https://www.7-zip.org/)

设置好git以准备更新github库
```bash
$ssh-keygen -t rsa -C "ElaineZhongXH@gmail.com" #将‪C:\Users\ACS1\.ssh\id_rsa.pub的内容复制到github的ssh key
$git config --global user.email "ElaineZhongXH@gmail.com"
$git config --global user.name "ElaineXHZhong"
# 常规步骤更新github库
$git add .
$git commit -m 'xxx'
$git push -u origin main
```

## Configure GPU Environment on Azure GPU VM

- VM GPU Info:
    ```markdown
    configuration:  24 vCPUs, 448G Memory and 4 NVIDIA Tesla V100 card
    Processor:      Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60 GHz 2.59 GHz (2 processors)
    Installed memory(RAM):  448 GB
    System type:    64-bit Operationg System, x64-based processor
    Windows edition:Windows Server 2019 Datacenter
    ```
- 下载安装NVIDIA Tesla (CUDA) drivers:  [Windows Server 2019 Driver](http://us.download.nvidia.com/tesla/451.82/451.82-tesla-desktop-winserver-2019-2016-international.exe) (.exe文件)
    ```markdown
    若运行需要gpu的程序出现: failed call to cuInit: CUDA_ERROR_NO_DEVICE
    则重新安装一次.exe文件即可(重装驱动)
    ```
- GPU配置过程同下
- 下载[lfw](http://vis-www.cs.umass.edu/lfw/lfw.tgz)数据集做测试
- 测试程序
    ```bash
    # (1) Align
    $python src/align/align_dataset_mtcnn.py datasets/lfw datasets/lfw_160 --image_size 160 --margin 32 
    # (2) Train
    $python src/classifier.py TRAIN datasets/lfw_160 models/20180402-114759/20180402-114759.pb models/lfw.pkl
    # (3) Classify
    $python src/classifier.py CLASSIFY datasets/lfw_160 models/20180402-114759/20180402-114759.pb models/lfw.pkl
    # (4) Predict
    $python contributed/predict.py datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg models/20180402-114759 models/lfw.pk
    ```


## Configure Conda Environment for tensorflow-gpu

配置GPU环境加速数据预处理和训练模型(配置CUDA,只有显卡为N卡才行):
```markdown
Visual Studio 2017 + python 3.6.12 + tensorflow-gpu 1.7.0 + CUDA 9.0 + cuDNN 7.0.5 + facenet site-packages
```
1. [Visual Studio 2017](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/)
2. [NVIDIA Driver](https://www.nvidia.cn/Download/index.aspx?lang=cn)
3. [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal)  
    - Base installer | Patch1 | Patch2 | Patch3 | Patch4
4. [cuDNN 7.0.5](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7)
    ```bash
    # 把cuDNN下的bin,include,lib文件夹拷贝到CUDA的安装路径: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
    # 要安的tensoflow版本不一样，所对应的CUDA 和cuDNN版本也就不一样 (一定要对应上，否则会报错)
    $nvcc -V   # 检查CUDA是否安装成功
    ```
5. 环境变量
    - 添加系统环境:
        ```markdown
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64
        C:\Program Files\NVIDIA Corporation\NVSMI   (nvidia-smi.exe的Path)
        C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common
        ```
    - 添加environment variables
        ```markdown
        CUDA_PATH       = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
        CUDA_PATH_V9_0  = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
        ```
6. [Anaconda 4.9.2](https://www.anaconda.com/)
    - 添加系统环境:
        ```markdown
        C:\ProgramData\Anaconda3
        C:\ProgramData\Anaconda3\Library\mingw-w64\bin
        C:\ProgramData\Anaconda3\Library\usr\bin
        C:\ProgramData\Anaconda3\Library\bin
        C:\ProgramData\Anaconda3\Scripts
        ```
    - 检查Anaconda是否安装成功
        ```bash
        $conda -V
        ```
7. install tensorflow-gpu 1.7.0
    ```bash
    $conda create -n facenet python=3.6 && conda activate facenet
    $cd facenet
    $pip install -r requirements.txt
    $pip uninstall -y tensorflow
    $pip install tensorflow-gpu==1.7.0
    ```
8. 配置site-packages下的facenet和align模块
    - 找到site-packages路径
        ```bash
        $conda activate facenet
        $where python # C:\Users\PC\.conda\envs\facenet\python.exe
        # site-packages就在C:\Users\PC\.conda\envs\facenet\Lib\site-packages
        ```
    - facenet模块
        ```markdown
        1. envs\facenet\Lib\site-packages下新建facenet文件夹
        2. 复制facenet-master\src下所有文件到上述facenet文件夹下
        3. 在conda环境中python, 然后import facenet, 不会报错即可
        ```
    - align模块
        ```markdown
        1. 复制facenet-master\src\align文件夹到envs\facenet\Lib\site-packages下
        2. 在conda环境中python, 然后import align, 不会报错即可
        ```
9. 调到2.5根据程序运行结果来添加缺失的包再继续接下来的步骤
10. 修改代码
    - `src\align\align_dataset_mtcnn.py`:
        ```python
        import facenet.facenet as facenet # 原: import facenet
        ```
    - `contributed\predict.py`
        ```python
        import facenet.facenet as facenet # 原: import facenet
        ```
11. 调整包版本
    ```bash
    $pip install numpy==1.16.2
    $pip install scipy==1.2.1
    # 后面启动server的时候还需要额外安装以下包
    $pip install pypinyin waitress imutils flask pillow
    # numpy如果不是指定版本，需要修改代码: numpy\lib\npyio.py: allow_pickle=False -> allow_pickle=True
    ```
12. 保持一致
    - 测试时的包版本要与训练模型时的包版本一致才可以预测(所以不要随便升级包版本)
        ```markdown
        AttributeError: 'SVC' object has no attribute '_probA' (or something like that)
        It turned out that I had to stay with the same version of scikit that was used to train the models I currently have. Later versions of scikit don't work with the trained face models. If you want to upgrade scikit, you have to retrain you models with the new version of scikit.
        ```
    - 训练和测试的基准模型保持一致
        ```markdown
        - 训练的时候使用的基准模型时models/20180402-114759/，predict的时候也请使用此模型
        - 训练的时候使用的基准模型时models/20180408-102900/，predict的时候也请使用此模型
        ```
13. 测试环境是否安装正确
    - 下载[lfw](http://vis-www.cs.umass.edu/lfw/)作测试
    ```bash
    # 1. Align
    $python src/align/align_dataset_mtcnn.py datasets/lfw datasets/lfw_160 --image_size 160 --margin 32 # sufficient GPU memory
    $python src/align/align_dataset_mtcnn.py datasets/lfw datasets/lfw_160 --image_size 160 --margin 32 --gpu_memory_fraction 0.5 # insufficient GPU memory强劲

    # 2. Train
    $python src/classifier.py TRAIN datasets/lfw_160 models/20180402-114759/20180402-114759.pb models/lfw.pkl

    # 3. Classify
    $python src/classifier.py CLASSIFY datasets/lfw_160 models/20180402-114759/20180402-114759.pb models/lfw.pkl

    # 4. Predict
    $python contributed/predict.py datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg models/20180402-114759 models/lfw.pkl # sufficient GPU memory
    $python contributed/predict.py datasets/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg models/20180402-114759 models/lfw.pkl --gpu_memory_fraction 0.5 # sufficient GPU memory
    ```
14. 如果需要查看时间
    ```python
    import time
    start_time  = time.time()
    # program
    print('Program Start At: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('Program End At: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('Program All Time: {0} seconds = {1} minutes = {2} hrs'.format((time.time() - start_time), (time.time() - start_time)/60, (time.time() - start_time)/3600))
    ```
