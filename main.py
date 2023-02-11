import time
import torch
from torch.backends import cudnn
import sys 
sys.path.append("HybridNets/")

from backbone import HybridNetsBackbone
import cv2
import numpy as np
from glob import glob
from utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params
from utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
from torchvision import transforms
import argparse
from utils.constants import *
from collections import OrderedDict
from torch.nn import functional as F


def main(args):
    params = Params(f'HybridNets/projects/{args.project}.yml')
    compound_coef = args.compound_coef
    source = args.source
    output = args.output
    weight = args.load_weights
    use_cuda = args.cuda
    use_float16 = args.float16

    anchors_ratios = params.anchors_ratios
    anchors_scales = params.anchors_scales
    threshold = args.conf_thresh
    iou_threshold = args.iou_thresh


    color_list_seg = {}
    for seg_class in params.seg_list:
        # edit your color here if you wanna fix to your liking
        color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))

    obj_list = params.obj_list
    seg_list = params.seg_list
    resized_shape = params.model['image_size']
    if isinstance(resized_shape, list):
        resized_shape = max(resized_shape)

    normalize = transforms.Normalize(
    mean=params.mean, std=params.std
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    weight = torch.load(weight, map_location='cuda' if use_cuda else 'cpu')
    weight_last_layer_seg = weight['segmentation_head.0.weight']
    if weight_last_layer_seg.size(0) == 1:
        seg_mode = BINARY_MODE
    else:
        if params.seg_multilabel:
            seg_mode = MULTILABEL_MODE
        else:
            seg_mode = MULTICLASS_MODE
    model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
                           scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=args.backbone,
                           seg_mode=seg_mode)
    model.load_state_dict(weight)

    model.requires_grad_(False)
    model.eval()

    kernel = np.ones((7,7),np.uint8)
    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()
    video_capture = cv2.VideoCapture(source)
    while True:
        ret, image = video_capture.read()
        if ret:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h0, w0 = image.shape[:2]  # orig hw
            r = resized_shape / max(h0, w0)  # resize image to img_size
            input_img = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
            h, w = input_img.shape[:2]

            (input_img, _), ratio, pad = letterbox((input_img, None), auto=False, scaleup=True)
            if use_cuda:
                x = torch.stack([transform(input_img).cuda()], 0)
            else:
                x = torch.stack([transform(input_img)], 0)
            x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)
            # x.unsqueeze_(0)
            shapes = []
            shapes.append(((h0, w0), ((h / h0, w / w0), pad)))
            with torch.no_grad():
                features, regression, classification, anchors, seg = model(x)
                # in case of MULTILABEL_MODE, each segmentation class gets their own inference image
                seg_mask_list = []
                # (B, C, W, H) -> (B, W, H)
                if seg_mode == BINARY_MODE:
                    seg_mask = torch.where(seg >= 0, 1, 0)
                    # print(torch.count_nonzero(seg_mask))
                    seg_mask.squeeze_(1)
                    seg_mask_list.append(seg_mask)
                elif seg_mode == MULTICLASS_MODE:
                    _, seg_mask = torch.max(seg, 1)
                    seg_mask_list.append(seg_mask)
                else:
                    seg_mask_list = [torch.where(torch.sigmoid(seg)[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]
                    # but remove background class from the list
                    seg_mask_list.pop(0)
                # (B, W, H) -> (W, H)
                for i in range(seg.size(0)):
                    for seg_class_index, seg_mask in enumerate(seg_mask_list):
                            seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
                            pad_h = int(shapes[i][1][1][1])
                            pad_w = int(shapes[i][1][1][0])
                            seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
                            seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)
                            color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
                            for index, seg_class in enumerate(params.seg_list):
                                if index == 1:
                                    mask = np.where(seg_mask_ == index+1 , 255, 0)
                                    mask = mask.astype(np.uint8)
                for i in range(3):
                    outmask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours, hierarchy = cv2.findContours(outmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                parameters = []
                for contour in contours:
                    [vx, vy, cx, cy]= cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    cv2.line(image, (int(cx-vx*w), int(cy-vy*w)), (int(cx+vx*w), int(cy+vy*w)), (0, 255, 0), 5)
                    coefficients = np.polyfit(np.array([int(cx-vx*w),int(cx+vx*w)]),np.array([int(cy-vy*w),int(cy+vy*w)]), 2)
                    parameters.append(coefficients)
                
                print(parameters)
        else:
                break





if __name__ == '__main__':

    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('--source', type=str, default='videos/NO20221023-112820-000013F.MP4', help='The demo image folder')
    parser.add_argument('--output', type=str, default='result', help='Output folder')
    parser.add_argument('-w', '--load_weights', type=str, default='weights/hybridnets.pth')
    parser.add_argument('--conf_thresh', type=restricted_float, default='0.75')
    parser.add_argument('--iou_thresh', type=restricted_float, default='0.5')
    parser.add_argument('--imshow', type=boolean_string, default=False, help="Show result onscreen (unusable on colab, jupyter...)")
    parser.add_argument('--imwrite', type=boolean_string, default=True, help="Write result to output folder")
    parser.add_argument('--show_det', type=boolean_string, default=False, help="Output detection result exclusively")
    parser.add_argument('--show_seg', type=boolean_string, default=False, help="Output segmentation result exclusively")
    parser.add_argument('--cuda', type=boolean_string, default=True)
    parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
    parser.add_argument('--speed_test', type=boolean_string, default=False,
                        help='Measure inference latency')
    args = parser.parse_args()
    main(args)
    