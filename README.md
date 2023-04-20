## Zhouyi Model Zoo

This repo provides a set of AI models for reference used by Zhouyi SDK.
| Summary               | Framework     | Input Shape                                 | Model Source                                                                                                   |
|-----------------------|---------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| age_googlenet         | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender                                     |
| alexnet               | caffe         | [1,3,227,227]                               | https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet                                                  |
| alexnet               | onnx          | [1,1,28,28]                                 | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| alexnet               | tflite(quant) | [1,28,28,1]                                 | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| alexnet               | tf            | [1,28,28,1]                                 | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| ann                   | onnx          | [1, 3, 1024, 2048]                          | https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ann                                           |
| arcface               | onnx          | [1, 3, 112, 112]                            | https://github.com/onnx/models/tree/master/vision/body_analysis/arcface                                        |
| bisenet_v2            | tf            | [4, 512, 1024, 3]                           | https://github.com/MaybeShewill-CV/bisenetv2-tensorflow                                                        |
| caffenet              | caffe         | [10, 3, 227, 227]                           | https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet                                       |
| caffenet              | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/classification/caffenet                                      |
| centerface            | onnx          | [1, 3, 640, 640]                            | https://github.com/ttanzhiqiang/onnx_tensorrt_project                                                          |
| centernet             | tf            | [1, 512, 512, 3]                            | https://hailo.ai/devzone-model-zoo/object-detection/                                                           |
| crnn_quant            | tflite(quant) | [1,64,200,3]                                | https://ai-benchmark.com/download.html                                                                         |
| deeplab_v2            | onnx          | [1, 3, 513, 513]                            | https://github.com/kazuto1011/deeplab-pytorch                                                                  |
| deeplab_v3            | tflite        | [1,513,513,3]                               | https://github.com/tensorflow/models/tree/archive/research/deeplab                                             |
| deeplab_v3            | onnx          | [1,3,513,513]                               | https://github.com/tensorflow/models/tree/archive/research/deeplab                                             |
| deeplab_v3            | tf            | [1,513,513,3]                               | https://github.com/tensorflow/models/tree/archive/research/deeplab                                             |
| deeplab_v3_plus_quant | tflite(quant) | [1,1024,1024,3]                             | https://ai-benchmark.com/download.html                                                                         |
| deepspeech_v2         | onnx          | [1,385,161,1]                               | https://github.com/tensorflow/models/tree/archive/research/deep_speech                                         |
| deepspeech_v2         | tf            | [1,385,161,1]                               | https://github.com/tensorflow/models/tree/archive/research/deep_speech                                         |
| densenet_121          | caffe         | [1, 3, 224, 224]                            | https://github.com/soeaver/caffe-model                                                                         |
| densenet_121          | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/classification/densenet-121                                  |
| densenet_169          | caffe         | [1, 3, 224, 224]                            | https://github.com/soeaver/caffe-model                                                                         |
| dilation_8            | caffe         | [1, 3, 900, 900]                            | https://github.com/fyu/dilation                                                                                |
| dped_quant            | tflite(quant) | [1,1536,2048,3]                             | https://ai-benchmark.com/download.html                                                                         |
| dpn_68_extra          | caffe         | [1, 3, 224, 224]                            | https://github.com/soeaver/caffe-model                                                                         |
| dpn_92                | caffe         | [1, 3, 224, 224]                            | https://github.com/soeaver/caffe-model                                                                         |
| duc                   | onnx          | [1, 3, 800, 800]                            | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/duc                            |
| efficientnet_b4_quant | tflite(quant) | [1,380,380,3]                               | https://ai-benchmark.com/download.html                                                                         |
| efficientnet_b5       | tf            | [1, 456, 456, 3]                            | https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet                                     |
| efficientnet_lite     | onnx          | [1,224,224,3]                               | https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4                            |
| enet                  | caffe         | [1, 3, 512, 1024]                           | https://github.com/TimoSaemann/Enet                                                                            |
| erfnet                | caffe         | [1,3,512,1024]                              | https://github.com/Yuelong-Yu/ERFNet-Caffe                                                                     |
| esrgan_quant          | tflite(quant) | [1,128,128,3]                               | https://ai-benchmark.com/download.html                                                                         |
| face_boxes            | onnx          | [1,3,1024,1024]                             | https://github.com/zisianw/FaceBoxes.PyTorch                                                                   |
| facenet               | tf            | [1, 160, 160, 3]                            | https://github.com/davidsandberg/facenet                                                                       |
| fast_depth            | onnx          | [1,3,224,224]                               | https://github.com/dwofk/fast-depth                                                                            |
| faster_rcnn           | caffe         | [1, 3, 224, 224]                            | https://github.com/rbgirshick/py-faster-rcnn                                                                   |
| fcn                   | caffe         | [1,3,224,224]                               | https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s-atonce                               |
| fcn16s                | onnx          | [1, 3, 224, 224]                            | https://github.com/wkentaro/pytorch-fcn/                                                                       |
| fcn32s                | onnx          | [1, 3, 224, 224]                            | https://github.com/wkentaro/pytorch-fcn/                                                                       |
| fcn8s                 | onnx          | [1, 3, 224, 224]                            | https://github.com/wkentaro/pytorch-fcn/                                                                       |
| googlenet             | caffe         | [10, 3, 224, 224]                           | https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet                                                |
| googlenet             | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet             |
| gru_l                 | tf            | [1, 49, 10]                                 | https://github.com/UT2UH/ML-KWS-for-ESP32/tree/master/Pretrained_models/GRU                                    |
| icnet                 | caffe         | [1, 3, 1025, 2049]                          | https://github.com/hszhao/ICNet                                                                                |
| imdn_quant            | tflite(quant) | [1,1024,1024,3]                             | https://ai-benchmark.com/download.html                                                                         |
| inception_resnet_v2   | tflite        | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| inception_resnet_v2   | tf            | [1, 299, 299, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| inception_resnet_v2   | caffe         | [1, 3,331,331]                              | https://github.com/soeaver/caffe-model                                                                         |
| inception_v1          | tflite(quant) | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| inception_v1          | tf            | [1, 224, 224, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| inception_v2          | tflite(quant) | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| inception_v2          | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2          |
| inception_v2          | tf            | [1, 224, 224, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| inception_v3          | tflite(quant) | [1, 299, 299, 3]                            | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| inception_v3          | tf            | [1, 299, 299, 3]                            | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| inception_v3          | caffe         | [1, 3, 299, 299]                            | https://github.com/soeaver/caffe-model/tree/master/cls                                                         |
| inception_v3          | onnx          | [1, 3, 299, 299]                            | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| inception_v3_quant    | tflite(quant) | [1, 346, 346, 3]                            | https://ai-benchmark.com/download.html                                                                         |
| inception_v4          | caffe         | [1,3,299,299]                               | https://github.com/soeaver/caffe-model/tree/master/cls                                                         |
| inception_v4          | onnx          | [1,3,299,299]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| inception_v4          | tflite(quant) | [1,299,299,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| inception_v4          | tf            | [1,299,299,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| lightface             | onnx          | [1, 3, 240, 320]                            | https://hailo.ai/devzone-model-zoo/face-detection/                                                             |
| lstm_quant            | tflite(quant) | [1,32,500,1]                                | https://ai-benchmark.com/download.html                                                                         |
| mask_rcnn             | tf            | [1,1024,1024,3]                             | https://github.com/matterport/Mask_RCNN                                                                        |
| mixnet                | tf            | [1,224,224,3]                               | https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mixnet-l                           |
| mnasnet               | tflite        | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| mobilebert_quant      | tflite(quant) | [1,384], [1,384], [1,384]                   | https://ai-benchmark.com/download.html                                                                         |
| mobilenet_edgetpu     | tflite        | [1,224,224,3]                               | https://github.com/mlcommons/mobile_models/blob/main/v0_7/tflite/mobilenet_edgetpu_224_1.0_uint8.tflite        |
| mobilenet_v1          | tflite(quant) | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| mobilenet_v1          | caffe         | [1, 3, 224, 224]                            | https://github.com/shicai/MobileNet-Caffe                                                                      |
| mobilenet_v1_224      | tf            | [1, 224, 224, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| mobilenet_v1_ssd      | tflite        | [300, 300]                                  | https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2                                          |
| mobilenet_v2          | caffe         | [1,3,224,224]                               | https://github.com/shicai/MobileNet-Caffe                                                                      |
| mobilenet_v2          | tflite(quant) | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets/mobilenet                                 |
| mobilenet_v2          | tf            | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets/mobilenet                                 |
| mobilenet_v2          | onnx          | [1,3,224,224]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets/mobilenet                                 |
| mobilenet_v2_b8_quant | tflite(quant) | [8,224,224,3]                               | https://ai-benchmark.com/download.html                                                                         |
| mobilenet_v2_quant    | tflite(quant) | [1,224,224,3]                               | https://ai-benchmark.com/download.html                                                                         |
| mobilenet_v2_ssd      | tf            | [1,300,300,3]                               | https://github.com/tensorflow/models/tree/archive/research/object_detection/models                             |
| mobilenet_v2_ssd      | onnx          | [1,300,300,3]                               | https://github.com/tensorflow/models/tree/archive/research/object_detection/models                             |
| mobilenet_v2_ssd      | caffe         | [1,3,300,300]                               | https://github.com/chuanqi305/MobileNet-SSD                                                                    |
| mobilenet_v2_ssd_lite | tf            | [1, 300, 300, 3]                            | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md          |
| mobilenet_v3          | tflite        | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet                                  |
| mobilenet_v3          | tf            | [1, 224, 224, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet                                  |
| mobilenet_v3_b4_quant | tflite(quant) | [4,512,512,3]                               | https://ai-benchmark.com/download.html                                                                         |
| mobilenet_v3_quant    | tflite(quant) | [1,512,512,3]                               | https://ai-benchmark.com/download.html                                                                         |
| mtcnn_o               | caffe         | [1,3,48,48]                                 | https://github.com/CongWeilin/mtcnn-caffe                                                                      |
| mtcnn_p               | caffe         | [1,3,12,12]                                 | https://github.com/CongWeilin/mtcnn-caffe                                                                      |
| mtcnn_r               | caffe         | [1,3,24,24]                                 | https://github.com/CongWeilin/mtcnn-caffe                                                                      |
| mv3_depth_quant       | tflite(quant) | [1,1024,1536,3]                             | https://ai-benchmark.com/download.html                                                                         |
| nasnet_large          | tf            | [1, 331, 331,3]                             | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| nasnet_mobile         | tflite        | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| nasnet_mobile         | tf            | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| peleenet              | caffe         | [1,3,224,224]                               | https://github.com/Robert-JunWang/PeleeNet/tree/master/caffe                                                   |
| pointnet              | onnx          | [32, 9, 4096]                               | https://github.com/yanx27/Pointnet_Pointnet2_pytorch                                                           |
| poly_lanenet          | onnx          | [1, 3, 640, 640]                            | https://hailo.ai/devzone-model-zoo/driving-lane-detection/                                                     |
| pspnet                | onnx          | [1, 3, 1024, 2048]                          | https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet                                        |
| punet_quant           | tflite(quant) | [1,544,960,4]                               | https://ai-benchmark.com/download.html                                                                         |
| pynet_quant           | tflite(quant) | [1,512,512,3]                               | https://ai-benchmark.com/download.html                                                                         |
| regnet_x              | onnx          | [1, 3, 224, 224]                            | https://hailo.ai/devzone-model-zoo/about-object-detection/                                                     |
| resnet_18             | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/main/vision/classification/resnet/model                                    |
| resnet_v1_101         | tflite(quant) | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| resnet_v1_101         | caffe         | [1,3,224,224]                               | https://github.com/SnailTyan/caffe-model-zoo                                                                   |
| resnet_v1_101         | onnx          | [1,3,224,224]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| resnet_v1_101         | tf            | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| resnet_v1_152         | tf            | [1, 224, 224, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| resnet_v1_50          | tflite(quant) | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| resnet_v1_50          | onnx          | [1,3,224,224]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| resnet_v1_50          | caffe         | [1,3,224,224]                               | https://github.com/SnailTyan/caffe-model-zoo                                                                   |
| resnet_v1_50          | tf            | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| resnet_v2_101         | tf            | [1, 299, 299, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| resnet_v2_101         | tflite        | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| resnet_v2_101         | caffe         | [1, 3, 224, 224]                            | https://github.com/soeaver/caffe-model                                                                         |
| resnet_v2_152         | tf            | [1, 299, 299, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| resnet_v2_152         | caffe         | [1, 3, 224, 224]                            | https://github.com/soeaver/caffe-model                                                                         |
| resnet_v2_50          | tf            | [1, 299, 299, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| resnext_101           | onnx          | [1, 3, 224, 224]                            | https://github.com/Cadene/pretrained-models.pytorch                                                            |
| resnext_101           | caffe         | [1, 3, 224, 224]                            | https://www.deepdetect.com/models/resnext/                                                                     |
| resnext_50            | caffe         | [1, 3, 224, 224]                            | https://github.com/soeaver/caffe-model                                                                         |
| rnn_t_decoder         | onnx          | [1, 1],[1, 2, 320],[1, 2, 320],[1, 1, 1024] | https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt                                     |
| rnn_t_encoder         | onnx          | [1,249,240]                                 | https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt                                     |
| se_inception          | caffe         | [1, 3, 224, 224]                            | https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/se-inception/README.md             |
| se_resnet_101         | caffe         | [1, 3, 224, 224]                            | https://github.com/hujie-frank/SENet                                                                           |
| se_resnet_50          | tf            | [1, 224, 224, 3]                            | https://github.com/HiKapok/TF-SENet                                                                            |
| se_resnext_50         | tf            | [1, 224, 224, 3]                            | https://github.com/HiKapok/TF-SENet                                                                            |
| shufflenet_v1         | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/classification/shufflenet                                    |
| shufflenet_v2         | caffe         | [1,3,224,224]                               | https://github.com/Ewenwan/ShuffleNet-2                                                                        |
| shufflenet_v2         | tflite(quant) | [1,224,224,3]                               | https://github.com/TropComplique/shufflenet-v2-tensorflow                                                      |
| shufflenet_v2         | tf            | [1,224,224,3]                               | https://github.com/TropComplique/shufflenet-v2-tensorflow                                                      |
| shufflenet_v2         | onnx          | [1,3,224,224]                               | https://github.com/TropComplique/shufflenet-v2-tensorflow                                                      |
| sne_roadseg           | onnx          | [1,3,384,1248]                              | https://github.com/hlwang1124/SNE-RoadSeg                                                                      |
| squeezenet            | caffe         | [10, 3, 227, 227]                           | https://github.com/forresti/SqueezeNet                                                                         |
| squeezenet            | tflite        | [1, 224, 224, 3]                            | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn                                                   |
| squeezenet_v1         | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/classification/squeezenet                                    |
| srcnn                 | tf            | [1, 33, 33, 1]                              | https://github.com/tegg89/SRCNN-Tensorflow                                                                     |
| srgan_quant           | tflite(quant) | [1,256,256,3]                               | https://ai-benchmark.com/download.html                                                                         |
| stacked_hourglass     | tf            | [1, 256, 256, 3]                            | https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras                                                |
| super_resolution      | onnx          | [1, 1, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016                          |
| swin_transformer      | onnx          | [1,3,384,384]                               | https://github.com/microsoft/Swin-Transformer                                                                  |
| unet_bio              | tf            | [1,256, 256,1]                              | https://github.com/zhixuhao/unet                                                                               |
| unet_quant            | tflite(quant) | [1,1024,1024,3]                             | https://ai-benchmark.com/download.html                                                                         |
| vgg_16                | onnx          | [1,3,224,224]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| vgg_16                | caffe         | [1,3,224,224]                               | https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md                                          |
| vgg_16                | tflite(quant) | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| vgg_16                | tf            | [1,224,224,3]                               | https://github.com/tensorflow/models/tree/archive/research/slim/nets                                           |
| vgg_19                | tf            | [1, 224, 224, 3]                            | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained                                      |
| vgg_cnn_s             | caffe         | [10, 3, 224, 224]                           | https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9                                                         |
| vgg_quant             | tflite(quant) | [1,256,256,1]                               | https://ai-benchmark.com/download.html                                                                         |
| vgg_ssd               | caffe         | [1, 3, 300, 300]                            | https://github.com/weiliu89/caffe/tree/ssd                                                                     |
| vsr_quant             | tflite(quant) | [1,540,960,3]                               | https://ai-benchmark.com/download.html                                                                         |
| wavenet               | onnx          | [1,390,23]                                  | https://github.com/buriburisuri/speech-to-text-wavenet                                                         |
| wavenet               | tf            | [1,390,23]                                  | https://github.com/buriburisuri/speech-to-text-wavenet                                                         |
| xception              | caffe         | [1, 3, 299, 299]                            | https://github.com/soeaver/caffe-model                                                                         |
| xlsr_quant            | tflite(quant) | [1,360,640,3]                               | https://ai-benchmark.com/download.html                                                                         |
| yolact_regnetx        | onnx          | [1, 3, 512, 512]                            | https://hailo.ai/devzone-model-zoo/instance-segmentation/                                                      |
| yolo_v2_416           | caffe         | [1,3,416,416]                               | https://github.com/tsingjinyun/caffe-yolov2                                                                    |
| yolo_v2_416           | tf            | [1,416,416,3]                               | https://github.com/wojciechmo/yolo2                                                                            |
| yolo_v2_416           | onnx          | [1,3,416,416]                               | https://github.com/wojciechmo/yolo2                                                                            |
| yolo_v3               | tf            | [1, 416, 416, 3]                            | https://github.com/qqwweee/keras-yolo3                                                                         |
| yolo_v3               | caffe         | [1, 3, 608, 608]                            | https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/tree/master/caffe_models/yolo_v3 |
| yolo_v4               | onnx          | [1, 416, 416, 3]                            | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4                         |
| yolo_v4_tiny_quant    | tflite(quant) | [1,416,416,3]                               | https://ai-benchmark.com/download.html                                                                         |
| yolo_v5               | onnx          | [1, 3, 640, 640]                            | https://github.com/ultralytics/yolov5                                                                          |
| zfnet_512             | onnx          | [1, 3, 224, 224]                            | https://github.com/onnx/models/tree/master/vision/classification/zfnet-512                                     |



&nbsp;



#### **FTP model download (Suggest FTP tool [FileZilla](https://filezilla-project.org/))**
  - `Host`: sftp://sftp01.armchina.com
  - `Account`:  zhouyi.armchina
  - `Password`:   114r3cJd 


&nbsp;

## About Zhouyi NPU
Zhouyi NPU IP is a high efficiency and high performance AI processor that supports wide range of AI application, including vision, audio/speech and NLP. Zhouyi NPU offers fixed function accelerator that speeds up general operations like convolution and activation, meanwhile Zhouyi NPU offers flexible programmability to support new operations or customized operations.

Zhouyi NPU IP has been deployed in a number of market segments, including surveillance, automotive, IOT, etc. With proven peformance and scalablility, Zhouyi NPU IP covers 0.25TOPS to hunderds of TOPS, target market from edge computing to data center.

&nbsp;

## About Zhouyi SDK
Zhouyi SDK is a full stack of platform that provides fast time-to-market development and deployment capability for user. 

![image](https://user-images.githubusercontent.com/85479712/198521602-49e13a31-bb49-424f-b782-5108274d63c3.png)

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
