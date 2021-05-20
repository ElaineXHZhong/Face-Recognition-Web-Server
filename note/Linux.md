# Linux Server

- <a href="#1">通过VSCode远程连接Linux服务器</a>
- <a href="#2">通过VSCode远程连接Windows服务器</a>
- <a href="#3">Linux服务器常规操作</a>
- <a href="#4">Windows服务器常规操作</a>

## <span id="1">通过VSCode远程连接Linux服务器</span>

**1. 本机安装OpenSSH**

Windows10下检查是否已经安装OpenSSH:
```bash
# Win + X -> 选择Windows PoweShell（管理员）
$Get-WindowsCapability -Online | ? Name -like 'OpenSSH*'
# 如果电脑未安装OpenSSH，则State会显示NotPresent | 安装了则是Installed
$Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# Win+R输入cmd进入终端
$ssh
```

**2. 本机安装Remote-SSH**

1. 打开VSCode左侧的插件搜索: SSH
2. 找到Remote-SSH直接安装即可

**3. 本机配置Remote-SSH**

1. 安装完成后，在VSCode左侧会出现一个远程资源管理器图标，点击
2. 左侧右边的SSH Targets右上方点击管理图标会出现一个输入框，进入config配置文件: `C:\Users\ACS1.ssh\config`
3. 在配置文件中设置服务器信息，输入`HostName`和`User`，保存以后左侧会出现对应机器名称
```bash
Host elaine # 随意机器名
    Hostname 10.6.36.231 # 服务器IP地址
    User kh # 用户名
```
4. 更改设置: File -> Preferences -> Settings -> Extension -> Remote-SSH
    - 找到Show Login Terminal并勾选

**4. 本机连接服务器**

1. 在左侧出现的对应机器名称的右侧有一个按钮，点击
2. 选择服务器的平台，并输入密码
3. 成功连上服务器，点击有右侧的+号创建服务器的终端窗口，可以正常使用了！

**5. 本机进入服务器的文件夹**

1. 左侧点击就可以看到 Open Folder，打开以后可以看到服务器文件目录
2. 直接在文件目录中选择文件进行编辑，实时同步到服务器上，这时候已经可以开始愉快的进行开发了，开发体验媲美本地开发！

## <span id="2">通过VSCode远程连接Windows服务器</span>

- <a href="#21">配置远程Windows服务器</a>
- <a href="#22">配置本机</a>

### <span id="21">配置远程Windows服务器</span>

**1. 远程服务器安装OpenSSH**

- 设置 -> 应用 -> 应用和功能 -> 管理可选功能
- 查看OpenSSH客户端是否已安装
- 如果没有，则在页面顶部选择“添加功能”: OpenSSH客户端 or OpenSSH 服务器

**2. SSH服务器的初始配置**

在搜索框中搜索powershell，单击“以管理员身份运行”，然后运行以下命令来启动 SSHD 服务：
```bash
$Start-Service sshd                                 # OPTIONAL but recommended:
$Set-Service -Name sshd -StartupType 'Automatic'    # Confirm the Firewall rule is configured. It should be created automatically by setup. 
$Get-NetFirewallRule -Name *ssh*                    # There should be a firewall rule named "OpenSSH-Server-In-TCP", which should be enabled
# If the firewall does not exist, create one
$New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 
```
找到Windows服务器的的username: C:\Users\username的username(`AstriACS`)

### <span id="22">配置本机</span>

**1. 启动PowerShell测试ssh连接**

```bash
# $Ssh username@servername    # <username> 是服务器 <server> 上的用户账号
$Ssh AstriACS@10.6.36.230
# 到任何服务器的第一个连接都将生成类似以下内容的消息
The authenticity of host servername (10.00.00.001) cant be established.
ECDSA key fingerprint is SHA256:(<a large string>).
Are you sure you want to continue connecting (yes/no)?
# 回答 Yes 会将该服务器添加到本地系统的已知 ssh 主机列表中
# 系统此时会提示你输入密码（也就是远程主机的开机密码）。 作为安全预防措施，密码在键入的过程中不会显示
# 在连接后，你将看到类似于以下内容的命令shell提示符: 
astriacs@DESKTOP-U1MHT5M C:\Users\AstriACS> # domain\username@SERVERNAME C:\Users\username>
```

**2. 在vscode中连接**

To connect to a remote host for the first time, follow these steps:
Verify you can connect to the SSH host by running the following command from a terminal / PowerShell window replacing user@hostname as appropriate.

```bash
$ssh user@hostname          # for Linux Server
$ssh user@domain@hostname   # Or for Windows when using a domain / AAD account 
```

打开vscode，打开Command Palette (F1/ctrl+shift+P) ，输入”Remote-SSH: Connect to Host…” 并选中，选择”+ Add New SSH Host…”，and use the same user@hostname as above，选择服务器的平台，并输入密码，可以正常使用了！

## <span id="3">Linux服务器常规操作</span>

- <a href="#31">查看系统信息</a>
- <a href="#32">查看GPU信息</a>
- <a href="#33">文件操作</a>

### <span id="31">查看系统信息</span>

```bash
$cat /proc/version              # 查看Linux内核版本命令
$uname -a                       # 查看Linux内核版本命令

$lsb_release -a                 # 查看Linux系统版本的命令: Ubuntu 20.04.1 LTS
$cat /etc/issue

$echo $PATH                     # 查看PATH环境变量
$vim /etc/profile               # 添加PATH环境变量
# 在文档最后，添加: export PATH="/opt/xxx/bin:$PATH"
# 保存，退出: Esc : wq
$source /etc/profile

$cd ~                           # 用户的主目录: /home/name
```

### <span id="32">查看GPU信息</span>

```bash
$sudo apt update                # 更新软件源
$sudo apt upgrade               # 升级软件包
$nvidia-smi                     # 查看支持的cuda版本, 如果无法查看，则说明尚未安装nvidia驱动

$uname -a                       # 查到Linux的架构是x86_64
$nvcc -V                        # 查看系统当前安装的CUDA的版本: Build cuda_11.0_bu.TC445_37.28540450_0
$cat /usr/local/cuda/version.txt # 如果nvcc没有安装，去安装目录下查看
$cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 # 查看 cuDNN 版本

$lspci | grep -i nvidia         # 查看NVIDIA GPU显卡信息
3b:00.0 3D controller: NVIDIA Corporation GV100GL [Tesla V100 PCIe 32GB] (rev a1)
af:00.0 3D controller: NVIDIA Corporation GV100GL [Tesla V100 PCIe 32GB] (rev a1)
d8:00.0 3D controller: NVIDIA Corporation GV100GL [Tesla V100 PCIe 32GB] (rev a1)
序号 3b:00.0, af:00.0, d8:00.0 是显卡的代号

$lspci -v -s 3b:00.0            # 查看指定显卡的详细信息
$nvidia-smi                     # 查看NVIDIA GPU显卡使用情况
$watch -n 10 nvidia-smi         # 周期性的输出显卡的使用情况, -n后边跟的是执行命令的周期，以s为单位
$sudo lshw -c video             # 查看显卡驱动
configuration: driver=nvidia latency=0

$apt search nvidia-cuda-toolkit # 查询目前可安装的CUDA Toolkit版本
```

```bash
# 运行facenet相关代码

# 选择1: 不用修改facenet代码 (CUDA 11.0 + tensorflow-gpu 1.14.0)
$pip install tensorflow-gpu==1.14.0 

# 选择2: 修改facenet代码     (CUDA 11.0 + tensorflow-gpu 2.4.0)
$pip install tensorflow-gpu==2.14.0  
# - 修改文件:
#     1. facenet/src/align/align_dataset_mtcnn.py
#     2. anaconda3/envs/facenet/lib/python3.6/site-packages/align/detect_face.py
# - 修改内容: 把所有的 tf. 替换为 tf.compat.v1.  (兼容性处理: tf.compat允许您编写在TensorFlow 1.x和2.x中均可使用的代码)

# - 修改文件: anaconda3/envs/facenet/lib/python3.6/site-packages/align/detect_face.py
# - 修改内容: 把第194行的 feed_in, dim = (inp, input_shape[-1].value) 替换为 feed_in, dim = (inp, input_shape[-1])
```

使用 PyTorch 查看 CUDA 和 cuDNN 版本
```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
```

### <span id="33">文件操作</span>

```bash
$unrar e filename.rar ../../All/        # 将 filename.rar 中的所有文件解压到../../All/目录下
$cp -r filename.rar dir2                # 将filename.rar移动到dir2目录下
$unzip filename.zip -d ../../All/       # 将 filename.zip 中的所有文件解压出来（-r 表示逐级压缩）
$unzip $(find . -name "*.zip") -d ../../Test/  # 找到当前目录下所有以".zip"结尾的文件并解压到../../Test/目录下

$find . -name "*.zip"                   # 查找目录下以为".zip"结尾的文件名（.可以换成其他目录的路径）
$find . -name "*.rar" 

$mkdir Time
$cd Time                               
$mkdir 20210511 20210518                # 创建两个文件夹

$mv $(find . -name "*.zip") ../Time/    # 找到当前目录下所有以".zip"结尾的文件并移动到../Time/目录下
$cd .. && rm -r 20210511                # 删除目录下的20210511文件夹及其子目录

$ls -d */                               # 列出此目录下所有的文件夹名
$ls -l *.png                            # 列出此目录下所有以".png"结尾的文件名
$ls -l *.mp4                            # 列出此目录下所有以".mp4"结尾的文件名

$wget url                               # 从指定的URL下载文件
```

## <span id="4">Windows服务器常规操作</span>

```bash
$mkdir Datasets                         # 创建Datasets文件夹

$cd ~/Downloads
$xcopy dir E:/Elaine/KOL/Datasets /e    # 复制文件夹到E:/Elaine/KOL/Datasets
$copy dir E:/Elaine/KOL/Datasets        # 复制文件到E:/Elaine/KOL/Datasets

$powershell (new-object Net.WebClient).DownloadFile('https://www.7-zip.org/a/7z1900-x64.exe','E:\Elaine\KOL\Software\7z.exe') # 利用powershell根据url下载文件并保存为7z.exe
```

