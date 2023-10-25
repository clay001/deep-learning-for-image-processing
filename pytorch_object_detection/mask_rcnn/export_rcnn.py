import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_segmentation_masks
from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import maskrcnn_resnet50_fpn

## 设置环境变量
## Mode 0：Modified MaskRCNN网络
## Mode C: Torchvision模型比较
## Mode E：导出onnx模型
## Mode V: 验证onnx图的正确性
## Mode NNA: 在CPU模拟NNA运行的标准结果
os.environ['Mode']='E'
plt.rcParams["savefig.bbox"] = 'tight'

# 创建two stage网络
def create_model(num_classes, load_pretrain_weights=True):
    from network_files import MaskRCNN
    from backbone import resnet50_fpn_backbone
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)
    # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d,
    #                                  trainable_layers=3)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)
    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        # for k in list(weights_dict.keys()):
        #     if ("box_predictor" in k) or ("mask_fcn_logits" in k):
        #         del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model

# 画图
def show(imgs, save_path=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

## 读入JPG文件，画原图
dog1_int = read_image(str('../../../dog_demo.jpg'))
dog_list = [dog1_int]
grid = make_grid(dog_list)
show(grid, "dog_origin")

# 模型创建
batch_int = torch.stack([dog1_int])
batch = convert_image_dtype(batch_int, dtype=torch.float)


model = create_model(num_classes=91, load_pretrain_weights=True)
model = model.eval()

if(os.environ['Mode']=='C'):
    model2 = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
    model2.eval()

    params1 = model.named_parameters()
    params2 = model2.named_parameters()
    for p1, p2 in zip(params1, params2):
        if p1[0] != p2[0]:
            print("name diff")
        if(torch.equal(p1[1], p2[1])):
            print(f"Parameters {p1[0]} are identical")
        else:
            import pdb;pdb.set_trace()

    # 写出输入的f32格式数据
    with open("dog.f32", "wb") as f:
        f.write(batch.numpy().tobytes())
    
    output2 = model2(batch)  

if(os.environ['Mode']=='E' ):
    torch.onnx.export(model, batch, "model.onnx", opset_version=11)
else:
    output = model(batch)

if(os.environ['Mode']=='V' ):
    import onnxruntime
    onnx_model_path = "/home/lockewang/Models/rcnn_infer/deep-learning-for-image-processing/pytorch_object_detection/mask_rcnn/model.onnx"
    # onnx_model_path = "/home/lockewang/Models/rcnn_infer/sim_model.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: batch.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)


    torch_outputs =list(output[0].values())

    for ort_output, torch_output in zip(ort_outputs, torch_outputs):
        np.testing.assert_allclose(ort_output, torch_output.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("Outputs match!")

    # transfer ort 2 output
    temp = dict()
    temp["boxes"] = torch.tensor(ort_outputs[0])
    temp["labels"] = torch.tensor(ort_outputs[1])
    temp["scores"] = torch.tensor(ort_outputs[2])
    temp["masks"] = torch.tensor(ort_outputs[3])
    output = [temp]

if(os.environ['Mode']=='E'):
     print("Exit programe for exporting!!")
     sys.exit ()
     
dog1_output = output[0]
dog1_masks = dog1_output['masks']
boxes = dog1_output['boxes'][:2]

# 打印mask的数据信息
print(f"shape = {dog1_masks.shape}, dtype = {dog1_masks.dtype}, "
      f"min = {dog1_masks.min()}, max = {dog1_masks.max()}")

inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}

print("For the first dog, the following instances were detected:")
print([inst_classes[label] for label in dog1_output['labels']])


## 画bounding box
## origin 0.8
score_threshold = .6
boxes = dog1_output['boxes'][dog1_output['scores'] > score_threshold]
colors = ["blue", "yellow"]
result = draw_bounding_boxes(dog1_int, boxes, colors=colors, width=5)
show(result, "dog_with_box")

# plot
proba_threshold = 0.5
dog1_bool_masks = dog1_output['masks'] > proba_threshold
print(f"shape = {dog1_bool_masks.shape}, dtype = {dog1_bool_masks.dtype}")

# There's an extra dimension (1) to the masks. We need to remove it
dog1_bool_masks = dog1_bool_masks.squeeze(1)

show(draw_segmentation_masks(dog1_int, dog1_bool_masks, alpha=0.9))

print(dog1_output['scores'])

#example 0.75
score_threshold = .6

boolean_masks = [
    out['masks'][out['scores'] > score_threshold] > proba_threshold
    for out in output
]

import pdb;pdb.set_trace()
dogs_with_masks = [
    draw_segmentation_masks(img, mask.squeeze(1))
    for img, mask in zip(batch_int, boolean_masks)
]
show(dogs_with_masks, "dog_mask")
