## Zhouyi Model Zoo

This repo provides a set of AI models for reference used by Zhouyi SDK.
| Summary | Model Source | Framework | Input Shape | Quant Model |
| - | - | - | - | - |
| faster_rcnn | https://github.com/rbgirshick/py-faster-rcnn | caffe | [1, 3, 224, 224] | No |
| mobilenet_v2 | https://github.com/tensorflow/models/tree/archive/research/slim/nets/mobilenet | tflite | [1,224,224,3] | No |
| Llama2-13b-gptq | https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ | onnx | [] | No |
| vgg_16 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tf1 | [1,224,224,3] | No |
| maskrcnn | https://github.com/matterport/Mask_RCNN | tf1 | [1,1024,1024,3] | No |
| yolo_v2_416 | https://github.com/wojciechmo/yolo2 | tf1 | [1,416,416,3] | No |
| Stable_Diffusion_V1.5 | https://huggingface.co/runwayml/stable-diffusion-v1-5 | onnx | [2,4,64,64],[1],[2,77,768] | No |
| Qwen1.5-1.8B | https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat | hf | [] | No |
| llama2_chn | https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf | hf | [1,256,4096], [1,1,256,512], [1,256] | No |
| llama2-7b | https://huggingface.co/meta-llama/Llama-2-7b-chat | hf | [] | No |
| ChatGLM3-6b | https://huggingface.co/THUDM/chatglm3-6b | hf | [1,256,4096], [1,1,256,512], [1,256] | No |
| Whisper_medium_en | https://huggingface.co/openai/whisper-medium.en/tree/main | onnx | encoder[[1,1,80,3000]] | No |
| deepspeech_v2 | https://github.com/tensorflow/models/tree/archive/research/deep_speech | onnx | [1,385,161,1] | No |
| deepspeech_v2 | https://github.com/tensorflow/models/tree/archive/research/deep_speech | tf1 | [1,385,161,1] | No |
| shufflenet_v2 | https://github.com/TropComplique/shufflenet-v2-tensorflow | tflite | [1,224,224,3] | No |
| mobilenet_v3 | https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet | tflite | [1,224,224,3] | No |
| yolo_v3_tiny | https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/tiny-yolov3 | onnx | [1,416,416,3] | No |
| yolo_v3 | https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/tree/master/caffe_models/yolo_v3 | caffe | [1, 3, 608, 608] | No |
| shufflenet_v2 | https://github.com/Ewenwan/ShuffleNet-2 | caffe | [1,3,224,224] | No |
| resnet_v1_50 | https://github.com/SnailTyan/caffe-model-zoo | caffe | [1,3,224,224] | No |
| mtcnn_p | https://github.com/CongWeilin/mtcnn-caffe | caffe | [1,3,12,12] | No |
| mtcnn_r | https://github.com/CongWeilin/mtcnn-caffe | caffe | [1,3,24,24] | No |
| mobilenet_v2_ssd | https://github.com/chuanqi305/MobileNet-SSD | caffe | [1,3,300,300] | No |
| mobilenet_v2 | https://github.com/tensorflow/models/tree/archive/research/slim/nets/mobilenet | tf1 | [1,224,224,3] | No |
| inception_v4 | https://github.com/soeaver/caffe-model/tree/master/cls | caffe | [1,3,299,299] | No |
| inception_v3 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tf1 | [1, 299, 299, 3] | No |
| inception_v3 | https://github.com/soeaver/caffe-model/tree/master/cls | caffe | [1, 3, 299, 299] | No |
| deeplab_v3 | https://github.com/tensorflow/models/tree/archive/research/deeplab | tf1 | [1,513,513,3] | No |
| vgg_16 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tflite | [1,224,224,3] | No |
| resnet_v1_101 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tflite | [1,224,224,3] | No |
| resnet_v1_50 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tflite | [1,224,224,3] | No |
| inception_v4 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tflite | [1,299,299,3] | No |
| alexnet | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tflite | [1,28,28,1] | No |
| inception_v3 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tflite | [1, 299, 299, 3] | No |
| yolo_v2_416 | https://github.com/wojciechmo/yolo2 | onnx | [1,3,416,416] | No |
| vgg_16 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | onnx | [1,3,224,224] | No |
| vgg_16 | https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md | caffe | [1,3,224,224] | No |
| shufflenet_v2 | https://github.com/TropComplique/shufflenet-v2-tensorflow | onnx | [1,3,224,224] | No |
| resnet_v1_101 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tf1 | [1,224,224,3] | No |
| resnet_v1_101 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | onnx | [1,3,224,224] | No |
| resnet_v1_101 | https://github.com/SnailTyan/caffe-model-zoo | caffe | [1,3,224,224] | No |
| mobilenet_v2_ssd | https://github.com/tensorflow/models/tree/archive/research/object_detection/models | onnx | [1,300,300,3] | No |
| mobilenet_v2 | https://github.com/tensorflow/models/tree/archive/research/slim/nets/mobilenet | onnx | [1,3,224,224] | No |
| inception_v4 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | onnx | [1,3,299,299] | No |
| inception_v3 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | onnx | [1, 3, 299, 299] | No |
| deeplab_v3 | https://github.com/tensorflow/models/tree/archive/research/deeplab | onnx | [1,3,513,513] | No |
| efficientnet_b5 | https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet | tf1 | [1, 456, 456, 3] | No |
| erfnet | https://github.com/Yuelong-Yu/ERFNet-Caffe | caffe | [1,3,512,1024] | No |
| deeplab_v3 | https://github.com/tensorflow/models/tree/archive/research/deeplab | tflite | [1,257,257,3] | No |
| alexnet | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tf1 | [1,28,28,1] | No |
| alexnet | https://github.com/tensorflow/models/tree/archive/research/slim/nets | onnx | [1,1,28,28] | No |
| alexnet | https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet | caffe | [1,3,227,227] | No |
| deeplab_v2 | https://github.com/kazuto1011/deeplab-pytorch | onnx | [1, 3, 513, 513] | No |
| yolo_v3 | https://github.com/qqwweee/keras-yolo3 | tf1 | [1, 416, 416, 3] | No |
| wavenet | https://github.com/buriburisuri/speech-to-text-wavenet | tf1 | [1,390,23] | No |
| mtcnn_o | https://github.com/CongWeilin/mtcnn-caffe | caffe | [1,3,48,48] | No |
| wavenet | https://github.com/buriburisuri/speech-to-text-wavenet | onnx | [1,390,23] | No |
| resnet_v1_50 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | onnx | [1,3,224,224] | No |
| peleenet | https://github.com/Robert-JunWang/PeleeNet/tree/master/caffe | caffe | [1,3,224,224] | No |
| facenet | https://github.com/davidsandberg/facenet | tf1 | [1, 160, 160, 3] | No |
| yolo_v2_416 | https://github.com/tsingjinyun/caffe-yolov2 | caffe | [1,3,416,416] | No |
| inception_v4 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tf1 | [1,299,299,3] | No |
| fcn | https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/voc-fcn8s-atonce | caffe | [1,3,224,224] | No |
| mobilenet_v2 | https://github.com/shicai/MobileNet-Caffe | caffe | [1,3,224,224] | No |
| shufflenet_v2 | https://github.com/TropComplique/shufflenet-v2-tensorflow | tf1 | [1,224,224,3] | No |
| resnet_v1_50 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | tf1 | [1,224,224,3] | No |
| mobilenet_v2_ssd | https://github.com/tensorflow/models/tree/archive/research/object_detection/models | tf1 | [1,300,300,3] | No |
| se_resnet_101 | https://github.com/hujie-frank/SENet | caffe | [1, 3, 224, 224] | No |
| mobilenet_v1_ssd | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md | tf1 | [1, 300, 300, 3] | No |
| centernet | https://hailo.ai/devzone-model-zoo/object-detection/ | tf1 | [1, 512, 512, 3] | No |
| arcface | https://github.com/onnx/models/tree/master/vision/body_analysis/arcface | onnx | [1, 3, 112, 112] | No |
| rnn_t_encoder | https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt | onnx | [1,249,240] | No |
| age_googlenet | https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender | onnx | [1, 3, 224, 224] | No |
| mobiledet_ssd | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md | tflite | [1, 320, 320, 3] | No |
| vgg_ssd | https://github.com/weiliu89/caffe/tree/ssd | caffe | [1, 3, 300, 300] | No |
| centerface | https://github.com/ttanzhiqiang/onnx_tensorrt_project | onnx | [1, 3, 640, 640] | No |
| enet | https://github.com/TimoSaemann/Enet | caffe | [1, 3, 512, 1024] | No |
| densenet | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 224, 224, 3] | No |
| efficientnet_lite | https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2 | tflite | [1, 300, 300, 3] | No |
| resnet_34_ssd | https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection | onnx | [1, 3, 1200, 1200] | No |
| efficientdet | https://tfhub.dev/tensorflow/lite-model/efficientdet/lite3/detection/default/1 | tflite | [ 1,512, 512, 3] | No |
| resnet_v1_ssd | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md | tf1 | [1, 640, 640, 3] | No |
| inception_v1 | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 224, 224, 3] | No |
| duc | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/duc | onnx | [1, 3, 800, 800] | No |
| yolo_v5 | https://github.com/ultralytics/yolov5 | tflite | [1, 640, 640, 3] | No |
| yolo_v5 | https://github.com/ultralytics/yolov5 | onnx | [1, 3, 640, 640] | No |
| mobilenet_v3 | https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet | tf1 | [1, 224, 224, 3] | No |
| yolo_v4 | https://github.com/hunglc007/tensorflow-yolov4-tflite | tf1 | [1,416,416,3] | No |
| ssd | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd | onnx | [1, 3, 1200, 1200] | No |
| yolox_s | https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime | onnx | [1, 3, 640, 640] | No |
| mobilenet_v1_ssd | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd-mobilenetv1 | onnx | [1, 3, 416, 416] | No |
| deeplabv3_mobilenetv2 | https://github.com/zym1119/DeepLabv3_MobileNetv2_PyTorch | onnx | [1, 3, 512, 512] | No |
| deeplab_v3_xception | https://github.com/tensorflow/models/blob/66264b2353aeeca3d4b340908a9590571495d5a6/research/deeplab/g3doc/model_zoo.md | tf1 | [1,513,513,3] | No |
| gender_googlenet | https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender | onnx | [1, 3, 224, 224] | No |
| efficientnet_l | https://hailo.ai/devzone-model-zoo/about-object-detection/ | tf1 | [1, 300, 300, 3] | No |
| inception_v2 | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 224, 224, 3] | No |
| mobilenet_v1 | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 224, 224, 3] | No |
| se_resnet_50 | https://github.com/soeaver/caffe-model | caffe | [1, 3, 225, 225] | No |
| zfnet_512 | https://github.com/onnx/models/tree/master/vision/classification/zfnet-512 | onnx | [1, 3, 224, 224] | No |
| se_resnet_50 | https://github.com/HiKapok/TF-SENet | tf1 | [1, 224, 224, 3] | No |
| apcnet | https://github.com/open-mmlab/mmsegmentation/tree/master/configs/apcnet | onnx | [1, 3, 1024, 2048] | No |
| ann | https://github.com/open-mmlab/mmsegmentation/tree/master/configs/ann | onnx | [1, 3, 1024, 2048] | No |
| se_inception | https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/se-inception/README.md | caffe | [1, 3, 224, 224] | No |
| pspnet | https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet | onnx | [1, 3, 1024, 2048] | No |
| yolact_regnetx | https://hailo.ai/devzone-model-zoo/instance-segmentation/ | onnx | [1, 3, 512, 512] | No |
| drn_38 | https://github.com/fyu/drn | onnx | [1, 3, 512, 1024] | No |
| yolo_v2 | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov2-coco | onnx | [1,  3, 416, 416] | No |
| yolo_v2_tiny | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2 | onnx | [1, 3, 416, 416] | No |
| vgg_cnn_s | https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9 | caffe | [10, 3, 224, 224] | No |
| vgg_19 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 224, 224, 3] | No |
| xception | https://github.com/soeaver/caffe-model | caffe | [1, 3, 299, 299] | No |
| unet_bio | https://github.com/zhixuhao/unet | tf1 | [1,256, 256,1] | No |
| srcnn | https://github.com/tegg89/SRCNN-Tensorflow | tf1 | [1, 33, 33, 1] | No |
| squeezenet | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 224, 224, 3] | No |
| squeezenet | https://github.com/onnx/models/tree/master/vision/classification/squeezenet | onnx | [1, 3, 224, 224] | No |
| shufflenet_v1 | https://github.com/onnx/models/tree/master/vision/classification/shufflenet | onnx | [1, 3, 224, 224] | No |
| sphereface | https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/Sphereface/README.md | caffe | [1, 3, 112, 96] | No |
| rnn_t_decoder | https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt | onnx | [1, 1],[1, 2, 320],[1, 2, 320],[1, 1, 1024] | No |
| retinanet | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/retinanet | onnx | [1, 3, 480, 640] | No |
| resnext_50 | https://github.com/soeaver/caffe-model | caffe | [1, 3, 224, 224] | No |
| resnext_101 | https://www.deepdetect.com/models/resnext/ | caffe | [1, 3, 224, 224] | No |
| resnet_v2_152 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 299, 299, 3] | No |
| resnet_v2_152 | https://github.com/soeaver/caffe-model | caffe | [1, 3, 224, 224] | No |
| resnet_v2_101 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 299, 299, 3] | No |
| resnet_v2_101 | https://github.com/soeaver/caffe-model | caffe | [1, 3, 448, 448] | No |
| resnet_v1_152 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 224, 224, 3] | No |
| poly_lanenet | https://hailo.ai/devzone-model-zoo/driving-lane-detection/ | onnx | [1, 3, 640, 640] | No |
| nasnet_mobile | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 224, 224, 3] | No |
| nasnet_mobile | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tf1 | [1, 224, 224, 3] | No |
| mobilenet_v2_ssd_lite | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md | tf1 | [1, 300, 300, 3] | No |
| mobilenet_v1_224 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 224, 224, 3] | No |
| mobilenet_v1 | https://github.com/shicai/MobileNet-Caffe | caffe | [1, 3, 224, 224] | No |
| mnasnet | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 224, 224, 3] | No |
| lightface | https://hailo.ai/devzone-model-zoo/face-detection/ | onnx | [1, 3, 240, 320] | No |
| mixnet | https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mixnet-l | tf1 | [1,224,224,3] | No |
| inception_v2_ssd | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md | tf1 | [1, 300, 300, 3] | No |
| inception_v2 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 224, 224, 3] | No |
| inception_v2 | https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2 | onnx | [1, 3, 224, 224] | No |
| inception_v1 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 224, 224, 3] | No |
| inception_resnet_v2 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 299, 299, 3] | No |
| inception_resnet_v2 | https://github.com/soeaver/caffe-model | caffe | [1, 3,331,331] | No |
| icnet | https://github.com/hszhao/ICNet | caffe | [1, 3, 1025, 2049] | No |
| googlenet | https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet | onnx | [1, 3, 224, 224] | No |
| googlenet | https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet | caffe | [10, 3, 224, 224] | No |
| fcn8s | https://github.com/wkentaro/pytorch-fcn/ | onnx | [1, 3, 224, 224] | No |
| fcn32s | https://github.com/wkentaro/pytorch-fcn/ | onnx | [1, 3, 224, 224] | No |
| fcn16s | https://github.com/wkentaro/pytorch-fcn/ | onnx | [1, 3, 224, 224] | No |
| fast_depth | https://github.com/dwofk/fast-depth | onnx | [1,3,224,224] | No |
| efficientnet_lite | https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4 | onnx | [1,224,224,3] | No |
| dpn_92 | https://github.com/soeaver/caffe-model | caffe | [1, 3, 224, 224] | No |
| efficientnet_lite | https://hailo.ai/devzone-model-zoo/about-object-detection/ | tf1 | [1, 280, 280, 3] | No |
| dpn_68_extra | https://github.com/soeaver/caffe-model | caffe | [1, 3, 224, 224] | No |
| densenet_169 | https://github.com/soeaver/caffe-model | caffe | [1, 3, 224, 224] | No |
| densenet_121 | https://github.com/onnx/models/tree/master/vision/classification/densenet-121 | onnx | [1, 3, 224, 224] | No |
| densenet_121 | https://github.com/soeaver/caffe-model | caffe | [1, 3, 224, 224] | No |
| caffenet | https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet | caffe | [10, 3, 227, 227] | No |
| bisenet_v2 | https://github.com/MaybeShewill-CV/bisenetv2-tensorflow | tf1 | [4, 512, 1024, 3] | No |
| resnet_v2_50 | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 299, 299, 3] | No |
| yolo_v5_s | https://github.com/ultralytics/yolov5/ | onnx | [1, 3, 640, 640] | No |
| yolo_v1_tiny | https://github.com/xingwangsfu/caffe-yolo | caffe | [1, 3, 448, 448] | No |
| super_resolution | https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016 | onnx | [1, 1, 224, 224] | No |
| resnet_v2_101 | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 299, 299, 3] | No |
| inception_v1 | https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v1 | onnx | [1, 3, 224, 224] | No |
| inception_resnet_v2 | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tflite | [1, 299, 299, 3] | No |
| face_boxes | https://github.com/zisianw/FaceBoxes.PyTorch | onnx | [1,3,1024,1024] | No |
| caffenet | https://github.com/onnx/models/tree/master/vision/classification/caffenet | onnx | [1, 3, 224, 224] | No |
| stacked_hourglass | https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras | tf1 | [1, 256, 256, 3] | No |
| squeezenet | https://github.com/forresti/SqueezeNet | caffe | [10, 3, 227, 227] | No |
| regnet_x | https://hailo.ai/devzone-model-zoo/about-object-detection/ | onnx | [1, 3, 224, 224] | No |
| yolo_v4 | https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4 | onnx | [1, 416, 416, 3] | No |
| densenet_201 | https://github.com/soeaver/caffe-model | caffe | [1, 3, 224, 224] | No |
| dilation_8 | https://github.com/fyu/dilation | caffe | [1, 3, 900, 900] | No |
| nasnet_large | https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn | tf1 | [1, 331, 331,3] | No |
| pnasnet_large | https://github.com/tensorflow/models/tree/master/research/slim#Pretrained | tf1 | [1, 331, 331, 3] | No |
| vision_transformer | https://github.com/jeonsworld/ViT-pytorch | onnx | [1,3,224,224] | No |
| resnext_101 | https://github.com/Cadene/pretrained-models.pytorch | onnx | [1, 3, 224, 224] | No |
| swin_transformer | https://github.com/microsoft/Swin-Transformer | onnx | [1,3,384,384] | No |
| gru_l | https://github.com/UT2UH/ML-KWS-for-ESP32/tree/master/Pretrained_models/GRU | tf1 | [1, 49, 10] | No |
| pointnet | https://github.com/yanx27/Pointnet_Pointnet2_pytorch | onnx | [32, 9, 4096] | No |
| fsrcnn | https://github.com/yjn870/FSRCNN-pytorch | onnx | [1, 1, 85, 85] | No |
| mobilenet_edgetpu | https://github.com/mlcommons/mobile_models/blob/main/v0_7/tflite/mobilenet_edgetpu_224_1.0_uint8.tflite | tflite | [1,224,224,3] | No |
| resnet_18 | https://github.com/onnx/models/tree/main/vision/classification/resnet/model | onnx | [1, 3, 224, 224] | No |
| codeformer_256 | https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0 | onnx | [1,3,256,256] | No |
| mobilenet_v3 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| inception_v3 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 299, 299, 3] | No |
| efficientnet_b5 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1,456,456,3] | No |
| mobilenet_v2 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| resnet_v1_101 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| resnet_v1_50 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| vgg_16 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| nasnet_mobile | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| densenet_121 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| resnet_v2_50 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| resnet_v2_101 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| vgg_19 | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 224, 224, 3] | No |
| xception | https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/applications | tf2 | [1, 299, 299, 3] | No |
| Qwen2.5-1.5B-Instruct-GPTQ | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 | hf | [] | No |
| Llama2-7b-gptq | https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ/tree/main | hf | [] | No |
| Qwen3-0.6B-GPTQ | https://huggingface.co/JunHowie/Qwen3-0.6B-GPTQ-Int4 | hf | [] | No |
| DeepSeek-R1-Distill-Qwen-7B_GPTQ | https://huggingface.co/jakiAJK/DeepSeek-R1-Distill-Qwen-7B_GPTQ-int4 | hf | [] | No |
| maskrcnn | https://pytorch.org/vision/main/models/mask_rcnn.html | pytorch | [3,800,800] | No |
| inception_v1 | https://github.com/onnx/models | onnx | [1,3,224,224] | Yes |
| Baichuan2-7b | https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat | onnx | [] | No |
| inception_resnet_v2 | https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionResNetV2 | tf2 | [1,299,299,3] | No |
| yolo_v8_seg | https://docs.ultralytics.com/tasks/segment/#models | onnx | [1, 3, 640, 640] | No |
| yolo_v5m | https://github.com/DefTruth/lite.ai.toolkit | onnx | [1,3,640,640] | No |
| yolo_v5s | https://github.com/ultralytics/yolov5/ | pytorch | [1,3,640,640] | No |
| yolo_v4_tiny | https://github.com/hunglc007/tensorflow-yolov4-tflite/tree/master | tflite | [1,416,416,3] | No |
| swin_transformer_base | https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md | onnx | [1,3,224,224] | No |
| MLP_EDSR | https://github.com/mlcommons/mobile_open/tree/main/vision/edsr | tflite | [1, 300, 500, 3] | No |
| MLP_mobiledet | https://github.com/mlcommons/mobile_open/tree/main/vision/mobiledet | tflite | [1, 320, 320, 3] | No |
| MLP_ssd_mobilenet_v2 | https://github.com/mlcommons/mobile_open/tree/main/vision/ssd_mobilenet | tflite | [1, 300, 300, 3] | No |
| MLP_deeplabV3_plus_MV2 | https://github.com/mlcommons/mobile_models/blob/main/v0_7/tflite/deeplabv3_mnv2_ade20k_int8.tflite | tflite | [1, 512, 512, 3] | No |
| resnet_34_ssd | https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection | tf1 | [1, 1200, 1200, 3] | No |
| yolox_l | https://github.com/Megvii-BaseDetection/YOLOX | onnx | [1,3,640,640] | No |
| sne_roadseg | https://github.com/hlwang1124/SNE-RoadSeg | onnx | [1,3,384,1248] | No |
| se_resnext_50 | https://github.com/HiKapok/TF-SENet | tf1 | [1, 224, 224, 3] | No |
| mobilenet_v1_ssd | https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2 | tflite | [300, 300] | No |
| sea_former | https://github.com/fudan-zvg/SeaFormer/tree/main/seaformer-seg | onnx | [1, 3, 512, 1024] | No |
| 3d_unet | https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19 | onnx | [1,1,128,128,128] | No |
| mosaic | https://tfhub.dev/google/lite-model/mosaic/mobilenetmultiavgseg/1 | tflite | [1, 1024, 2048, 3] | No |
| yolo_v6s | https://github.com/DefTruth/lite.ai.toolkit | onnx | [1,3,640,640] | No |
| transformer_official | https://github.com/Kyubyong/transformer | tf1 |  [1, 32] | No |
| ViT_B_16 | https://pytorch.org/vision/stable/models/vision_transformer.html | pytorch | [1, 3, 224, 224] | No |
| efficientnet_b4_quant | https://ai-benchmark.com/download.html | tflite | [1,380,380,3] | Yes |
| dped_quant | https://ai-benchmark.com/download.html | tflite | [1,1536,2048,3] | Yes |
| shufflenet_v2 | https://github.com/TropComplique/shufflenet-v2-tensorflow | onnx | [1,3,224,224] | Yes |
| vgg_quant | https://ai-benchmark.com/download.html | tflite | [1,256,256,1] | Yes |
| caffenet | https://github.com/onnx/models/tree/master/vision/classification/caffenet | onnx | [1, 3, 224, 224] | Yes |
| mobilebert_quant | https://ai-benchmark.com/download.html | tflite | [1,384], [1,384], [1,384] | Yes |
| shufflenet_v2 | https://pytorch.org/vision/stable/models/shufflenetv2.html | pytorch | [1, 3, 224, 224] | Yes |
| fcn32s | https://github.com/wkentaro/pytorch-fcn/ | onnx | [1, 3, 224, 224] | Yes |
| imdn_quant | https://ai-benchmark.com/download.html | tflite | [1,1024,1024,3] | Yes |
| squeezenet | https://github.com/onnx/models/tree/master/vision/classification/squeezenet | onnx | [1, 3, 224, 224] | Yes |
| resnet_v1_50 | https://pytorch.org/vision/stable/models/resnet.html | pytorch | [1, 3, 224, 224] | Yes |
| vsr_quant | https://ai-benchmark.com/download.html | tflite | [1,540,960,3] | Yes |
| punet_quant | https://ai-benchmark.com/download.html | tflite | [1,544,960,4] | Yes |
| dped_instance_quant | https://ai-benchmark.com/download.html | tflite | [1,1024,1536,3] | Yes |
| mobilenet_v2_quant | https://ai-benchmark.com/download.html | tflite | [1,224,224,3] | Yes |
| vgg_16 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | onnx | [1,3,224,224] | Yes |
| mobilenet_v2_b8_quant | https://ai-benchmark.com/download.html | tflite | [8,224,224,3] | Yes |
| unet_quant | https://ai-benchmark.com/download.html | tflite | [1,1024,1024,3] | Yes |
| resnet_v1_50 | https://github.com/tensorflow/models/tree/archive/research/slim/nets | onnx | [1,3,224,224] | Yes |
| mobilenet_v3_quant | https://ai-benchmark.com/download.html | tflite | [1,512,512,3] | Yes |
| crnn_quant | https://ai-benchmark.com/download.html | tflite | [1,64,200,3] | Yes |
| deeplab_v3_plus_quant | https://ai-benchmark.com/download.html | tflite | [1,1024,1024,3] | Yes |
| age_googlenet | https://github.com/onnx/models/tree/master/vision/body_analysis/age_gender | onnx | [1, 3, 224, 224] | Yes |
| googlenet | https://pytorch.org/vision/stable/models/googlenet.html | pytorch | [1, 3, 224, 224] | Yes |
| inception_v3 | https://pytorch.org/vision/stable/models/generated/torchvision.models.quantization.inception_v3.html#torchvision.models.quantization.inception_v3 | pytorch | [1,3,224,224] | Yes |
| srgan_quant | https://ai-benchmark.com/download.html | tflite | [1,256,256,3] | Yes |
| mv3_depth_quant | https://ai-benchmark.com/download.html | tflite | [1,1024,1536,3] | Yes |
| mobilenet_v3 | https://pytorch.org/vision/stable/models/mobilenetv3.html | pytorch | [1, 3, 224, 224] | Yes |
| mobiledet_ssd | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md | tflite | [1, 320, 320, 3] | Yes |
| pynet_quant | https://ai-benchmark.com/download.html | tflite | [1,512,512,3] | Yes |
| mobilenet_v2 | https://github.com/tensorflow/models/tree/archive/research/slim/nets/mobilenet | onnx | [1,3,224,224] | Yes |
| lstm_quant | https://ai-benchmark.com/download.html | tflite | [1,32,500,1] | Yes |
| yolo_v4_tiny_quant | https://ai-benchmark.com/download.html | tflite | [1,416,416,3] | Yes |
| inception_v3_quant | https://ai-benchmark.com/download.html | tflite | [1, 346, 346, 3] | Yes |
| esrgan_quant | https://ai-benchmark.com/download.html | tflite | [1,128,128,3] | Yes |
| mobilenet_v3_b4_quant | https://ai-benchmark.com/download.html | tflite | [4,512,512,3] | Yes |
| xlsr_quant | https://ai-benchmark.com/download.html | tflite | [1,360,640,3] | Yes |
| efficientnet_b5 | https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b5.html#torchvision.models.efficientnet_b5 | pytorch | [1, 3, 456, 456] | No |
| petr | https://github.com/megvii-research/PETR | pytorch | [6, 3, 320, 800] | No |
| fcn | https://pytorch.org/vision/stable/models/fcn.html | pytorch | [1, 3, 224, 224] | No |
| centernet_resnet50 | https://github.com/bubbliiiing/centernet-pytorch | pytorch | [1, 3, 512, 512] | No |
| yolo_v5s6 | https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt | onnx | [1, 3, 640, 640] | No |
| shufflenet_v2 | https://pytorch.org/vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_0.html#torchvision.models.shufflenet_v2_x1_0 | pytorch | [1,3,224,224] | No |
| yolo_v4 | https://github.com/bubbliiiing/yolov4-pytorch/tree/master | pytorch | [1, 3, 416, 416] | No |
| resnet_50_v1_5 | https://zenodo.org/ | tf1 | [1, 224, 224, 3] | No |
| alexnet | https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html#torchvision.models.alexnet | pytorch | [2,3,224,224] | No |
| yolo_v4_tiny | https://github.com/bubbliiiing/yolov4-tiny-pytorch | pytorch | [1, 3, 416, 416] | No |
| vgg_16 | https://pytorch.org/vision/stable/models/vgg.html | pytorch | [1, 3, 224, 224] | No |
| mobilenet_v1_224 | https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md | tflite | [1, 224, 224, 3] | No |
| swin_transformer_tiny_224 | https://pytorch.org/vision/stable/models/swin_transformer.html | pytorch | [1, 3, 224, 224] | No |
| resnet_v1_50 | https://pytorch.org/vision/stable/models/resnet.html | pytorch | [1, 3, 224, 224] | No |


&nbsp;



#### **FTP model download (Suggest FTP tool [FileZilla](https://filezilla-project.org/))**
 - `Host`: sftp://sftp01.armchina.com
 - `Account`: zhouyi.armchina
 - `Password`: 114r3cJd 


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
