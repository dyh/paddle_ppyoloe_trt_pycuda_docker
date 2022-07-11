# pp-yoloe -> tensorrt 部署 -> pycuda 推理

## 项目信息

paddlepaddle pp-yoloe -> onnx -> tensorrt 部署 -> pycuda 推理


## 视频

bilibili

[![bilibili](https://raw.githubusercontent.com/dyh/paddle_ppyoloe_trt_pycuda_docker/main/cover1.jpg)](https://www.bilibili.com/video/BV1R34y1n71w/ "bilibili")



## 运行环境

- 环境复杂，直接上Docker
- all you need is Dockerfile


## 宿主机环境

- 操作系统 ubuntu 22.04
- 显卡型号 rtx3090
- 显卡驱动 515.48.07
- docker版本 20.10.17
- nvidia-container-toolkit版本 2.11.0-1


## 如何运行

1. 安装 docker

    https://docs.docker.com/engine/install/ubuntu/


    ```
    $ sudo apt-get update

    $ sudo apt-get install \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    $ sudo mkdir -p /etc/apt/keyrings

    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    $ echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    $ sudo apt-get update

    $ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
    ```


2. 安装 nvidia-container-toolkit

    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker


    ```
    $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
          && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
          && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    $ sudo apt-get update

    $ sudo apt-get install -y nvidia-docker2

    $ sudo systemctl restart docker

    $ sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
    ```


3. 下载代码

    ```
    $ git clone https://github.com/dyh/paddle_ppyoloe_trt_pycuda_docker.git
    ```

4. 进入 Dockerfile 目录

    ```
    $ cd paddle_ppyoloe_trt_pycuda_docker/dockerfile
    ```

5. 编译 docker image

    > 切换到 root 账户

    ```
    $ su

    $ docker build -t trt/ppyoloe:v1 .
    ```

6. 创建 container

    > 挂载目录

    ```
    $ cd ../

    $ docker run -itd --gpus all -v ${PWD}:/TRT --name trt_1 trt/ppyoloe:v1
    ```


7. 进入 container

    ```
    $ docker exec -it trt_1 bash
    ```


8. 安装 PaddlePaddle

    ```
    $ python -m pip install paddlepaddle-gpu==2.3.1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
    ```


9. 安装 PaddleDetection

    ```
    $ cd /TRT

    $ wget https://github.com/PaddlePaddle/PaddleDetection/archive/refs/tags/v2.4.0.zip

    $ unzip v2.4.0.zip

    $ cd PaddleDetection-2.4.0

    $ pip install -r requirements.txt && python setup.py install

    $ pip uninstall opencv-python -y && \
        pip install opencv-python-headless && \
        pip uninstall opencv-python-headless -y && \
        pip install opencv-python-headless

    ```


10. 导出 paddle 权重 pdparams

    ```
    $ python tools/export_model.py \
             -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
             -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams \
             trt=True \
             exclude_nms=True \
             TestReader.inputs_def.image_shape=[3,640,640] \
             --output_dir ./
    ```

11. pdparams -> ONNX

    ```
    $ paddle2onnx --model_dir ./ppyoloe_crn_s_300e_coco \
                  --model_filename model.pdmodel \
                  --params_filename model.pdiparams \
                  --save_file out_s.onnx \
                  --input_shape_dict "{'image':[1, 3, 640, 640], 'scale_factor': [1, 2]}"
    ```


12. 添加 EfficientNMS 算子 -> NMS ONNX

    ```
    $ python ../EfficientNMS_TRT.py \
             --weights_type s \
             --input_path out_s.onnx \
             --save_path out_s_nms.onnx \
             --class_num 80 \
             --score_threshold 0.5 \
             --iou_threshold 0.4
    ```

13. NMS ONNX -> TensorRT .engine

    ```
    $ /usr/src/tensorrt/bin/trtexec --onnx=./out_s_nms.onnx --saveEngine=./out_s_fp16.engine --fp16
    ```


14. 使用 pycuda 调用 .engine 权重，推理 ./images/bus.jpg

    ```
    $ cd ..
    $ python infer_pycuda.py
    ```

15. 查看结果

    ```
    ./bus.jpg
    ```


## 项目引用

https://github.com/Monday-Leo/PPYOLOE_Tensorrt

https://github.com/NVIDIA/TensorRT/blob/main/docker/ubuntu-20.04.Dockerfile
