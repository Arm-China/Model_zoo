## Zhouyi Model Zoo

This repo provides configuration file for build tool/NN-compiler usage by Zhouyi SDK.
>**Note**: For all the models listed as below, frozen models are put in SFTP server. 
&nbsp;
### Classification
- Resnet-v1-50
  > [TF model](https://github.com/tensorflow/models/tree/archive/research/slim/nets),
    [Caffe model](https://github.com/SnailTyan/caffe-model-zoo)
- [Resnet-v2-50](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
- Resnet-v1-101
  > [TF model](https://github.com/tensorflow/models/tree/archive/research/slim/nets),
    [Caffe model](https://github.com/SnailTyan/caffe-model-zoo)
- [Resnet-v2-101](https://github.com/soeaver/caffe-model)
- [Resnet-v1-152](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
- [Resnet-v2-152](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
- [Resnext-50](https://github.com/soeaver/caffe-model)
- [VGG-19](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
- [Inception-v1](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
- inceptiont-v2
  > [TF model](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)
  [onnx_model](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2)
- [Xception](https://github.com/soeaver/caffe-model)
- Mobilenet-v1
  > [TF model](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained),
  [Caffe model](https://github.com/shicai/MobileNet-Caffe)
- [mobilenet_v3](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
- [Squeezenet](https://github.com/forresti/SqueezeNet)
- [densenet_121](https://github.com/soeaver/caffe-model)
- [dilation8](https://github.com/fyu/dilation)
- [shufflenet_v1](https://github.com/onnx/models/tree/master/vision/classification/shufflenet)
- [Densenet-169](https://github.com/soeaver/caffe-model)
- [DPN68-extra](https://github.com/soeaver/caffe-model)

### Detection
- [mobilenet_v1_ssd](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
- [VGG-SSD](https://github.com/weiliu89/caffe/tree/ssd)
- [YOLO-v3](https://github.com/qqwweee/keras-yolo3)
- [PFLD](https://github.com/guoqiangqi/PFLD)

### Segmentation
- [fcn32s](https://github.com/wkentaro/pytorch-fcn/)
- [fcn16s](https://github.com/wkentaro/pytorch-fcn/)
- [fcn8s](https://github.com/wkentaro/pytorch-fcn/)
- UNet
- [ENet](https://github.com/TimoSaemann/Enet)
- [ICNet](https://github.com/hszhao/ICNet)
- [unet_bio](https://github.com/zhixuhao/unet)

### Super-resolution
- [srcnn](https://github.com/tegg89/SRCNN-Tensorflow)

### Pose estimation
- [centerpose](https://hailo.ai/devzone-model-zoo/pose-estimatiom/)

&nbsp;
#### **FTP model download (Suggest FTP tool [FileZilla](https://filezilla-project.org/))**
  - `Host`: sftp://sftp01.armchina.com
  - `Account`:  zhouyi.armchina
  - `Password`:   114r3cJd 
&nbsp;

## About Zhouyi NPU
Zhouyi NPU IP is a high efficiency and high performance AI processor that supports wide range of AI application, including vision, audio/speech and NLP. Zhouyi NPU offers fixed function accelerator that speeds up general operations like convolution and activation, meanwhile Zhouyi NPU offers flexible programmability to support new operations or customized operations.

Zhouyi NPU IP has been deployed in a number of market segments, including surveillance, automotive, IOT, etc. With proven peformance and scalablility, Zhouyi NPU IP covers 0.25TOPS to hunders of TOPS, target market from edge computing to data center.
&nbsp;

## About Zhouyi SDK
Zhouyi SDK is a full stack of platform that provides fast time-to-market development and deployment capability for user. 
![imgae](zhouyi_toolkit.jpg)

### Highlighted feature:
- Support various framework
  > Supports TensorFlow, TensorFlow Lite, Caffe, Pytorch and ONNX, extend to support orther framework by IR transition

- Graph optimization
  > Many graph optimization techniques are used to reduce network complexity, like conv+batchnorm+relu fusion, to make graph run fast and efficiently on Zhouyi hardware. 

- Quantization
  > Support training-aware quantization model and post-training quantization method. Natively provides advanced-quantization techniques, also provide interface for user customized quantization parameter.

- IR (intermediate representation)
  > Public Zhouyi IR, and can be a standard for other framework to support

- Customization
  > Provide plugin interface for user easy to develop custom operations and integrate into SDK

- Simulator 
  > Provide instruction-level simulator, easy for user to evaluate model before porting to hardware
