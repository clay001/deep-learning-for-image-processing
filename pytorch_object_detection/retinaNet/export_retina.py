import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import retinanet_resnet50_fpn


from backbone import resnet50_fpn_backbone, LastLevelP6P7
from network_files import RetinaNet

## 设置环境变量
## Mode 0：Modified Retina网络
## Mode C: Torchvision模型比较
## Mode E：导出onnx模型
## Mode V: 验证onnx图的正确性
## Mode NNA: 在CPU模拟NNA运行的标准结果
os.environ['Mode']='V'
plt.rcParams["savefig.bbox"] = 'tight'

# 创建one stage网络
def create_model(num_classes):
    # 创建retinanet_res50_fpn模型
    # skip P2 because it generates too many anchors (according to their paper)
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256),
                                     trainable_layers=3)
    model = RetinaNet(backbone, num_classes)

    # 载入预训练权重
    # https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
    weights_dict = torch.load("./retinanet_resnet50_fpn.pth", map_location='cpu')
    # 删除分类器部分的权重，因为自己的数据集类别与预训练数据集类别(91)不一定致，如果载入会出现冲突
    # del_keys = ["head.classification_head.cls_logits.weight", "head.classification_head.cls_logits.bias"]
    # for k in del_keys:
    #     del weights_dict[k]
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
kobe_int = read_image(str('../../../../retina_infer/pascal.jpg'))
# kobe_int = kobe_int[:,:800,100:900]
kobe_list = [kobe_int]
grid = make_grid(kobe_list)
show(grid, "kobe_origin")

# 模型创建
batch_int = torch.stack([kobe_int])
batch = convert_image_dtype(batch_int, dtype=torch.float)


model = retinanet_resnet50_fpn(pretrained=True, progress=False)#
# model = create_model(num_classes=91)
model = model.eval()

if(os.environ['Mode']=='C'):
    model2 = retinanet_resnet50_fpn(pretrained=True, progress=False)
    model2.eval()

    params1 = model.state_dict()
    params2 = model2.state_dict()
    
    name_list_1 = list(params1.keys())
    name_list_2 = list(params2.keys())
    not_the_same = []
    addition = []
    for i in range(len(name_list_1)):
        key = name_list_1[i]
        if key not in name_list_2:
            addition.append(key)
            continue
        p1 = params1[key]
        p2 = params2[key]
        if(torch.equal(p1, p2)):
            print(f"Parameters {key} are identical")
        else:
            not_the_same.append(key)
    
    import pdb; pdb.set_trace()
    # 写出输入的f32格式数据
    with open("kobe.f32", "wb") as f:
        f.write(batch.numpy().tobytes())
    
    output2 = model2(batch)  

if(os.environ['Mode']=='E' ):
    torch.onnx.export(model, batch, "model.onnx", opset_version=11)
else:
    output = model(batch)

if(os.environ['Mode']=='V' ):
    import onnxruntime
    # onnx_model_path = "/home/lockewang/Models/retina_infer/resnext50_32x4d_fpn.onnx"
    onnx_model_path = "./model.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: batch.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)


    torch_outputs =list(output[0].values())
    import pdb;pdb.set_trace()
    for ort_output, torch_output in zip(ort_outputs, torch_outputs):
        np.testing.assert_allclose(ort_output, torch_output.detach().numpy(), rtol=1e-4, atol=1e-4)
    print("Outputs match!")

    # transfer ort 2 output
    temp = dict()
    temp["boxes"] = torch.tensor(ort_outputs[0])
    temp["labels"] = torch.tensor(ort_outputs[2])
    temp["scores"] = torch.tensor(ort_outputs[1])
    output = [temp]

if(os.environ['Mode']=='E'):
     print("Exit programe for exporting!!")
     sys.exit ()
     
kobe_output = output[0]

# # 打印mask的数据信息
# print(f"shape = {dog1_masks.shape}, dtype = {dog1_masks.dtype}, "
#       f"min = {dog1_masks.min()}, max = {dog1_masks.max()}")

# inst_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
#                 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
#                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # class names
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

print("For the first kobe, the following instances were detected:")
print([inst_classes[label] for label in kobe_output['labels']])


## 画bounding box
## origin 0.8
score_threshold = .4
boxes = kobe_output['boxes'][kobe_output['scores'] > score_threshold]
colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple'] # 红橙黄绿青蓝紫
result = draw_bounding_boxes(kobe_int, boxes, colors=colors, width=5)
show(result, "kobe_with_box")

print(kobe_output['scores'])