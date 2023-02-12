import sys

import torch

sys.path.append("HybridNets/")

import argparse
import os

import cv2
import numpy as np
from backbone import HybridNetsBackbone
from torchvision import transforms
from utils.constants import *
from utils.utils import (BBoxTransform, ClipBoxes, Params, boolean_string,
                         letterbox, postprocess, restricted_float,
                         scale_coords)
import warnings
warnings.simplefilter('ignore', np.RankWarning)

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
    frame_Counter = 0
    while True:
        ret, image = video_capture.read()
        frame_Counter += 1
        if ret:

            h0, w0 = image.shape[:2]  
            r = resized_shape / max(h0, w0)  
            input_img = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
            h, w = input_img.shape[:2]

            (input_img, _), ratio, pad = letterbox((input_img, None), auto=False, scaleup=True)
            if use_cuda:
                x = torch.stack([transform(input_img).cuda()], 0)
            else:
                x = torch.stack([transform(input_img)], 0)
            x = x.to(torch.float16 if use_cuda and use_float16 else torch.float32)
            shapes = []
            shapes.append(((h0, w0), ((h / h0, w / w0), pad)))
            with torch.no_grad():
                _, _, _, _, seg = model(x)
                seg_mask_list = []

                if seg_mode == BINARY_MODE:
                    seg_mask = torch.where(seg >= 0, 1, 0)

                    seg_mask.squeeze_(1)
                    seg_mask_list.append(seg_mask)
                elif seg_mode == MULTICLASS_MODE:
                    _, seg_mask = torch.max(seg, 1)
                    seg_mask_list.append(seg_mask)
                else:
                    seg_mask_list = [torch.where(torch.sigmoid(seg)[:, i, ...] >= 0.5, 1, 0) for i in range(seg.size(1))]
                    seg_mask_list.pop(0)

                for i in range(seg.size(0)):
                    for seg_mask in seg_mask_list:
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

                outmask = mask.copy()
                for i in range(2):
                    outmask = cv2.morphologyEx(outmask, cv2.MORPH_OPEN, kernel)
                    outmask = cv2.erode(outmask, kernel)

                all_contours, _ = cv2.findContours(outmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = 2000
                contours = [cnt for cnt in all_contours if cv2.contourArea(cnt) > min_area]
                parameters = []
                for contour in contours:
                    [vx, vy, cx, cy]= cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    cv2.line(image, (int(cx-vx*w0), int(cy-vy*w0)), (int(cx+vx*w0), int(cy+vy*w0)), (0, 255, 0), 5)
                    coefficients = np.polyfit(np.array([int(cx-vx*w0),int(cx+vx*w0)]),np.array([int(cy-vy*w0),int(cy+vy*w0)]), 2)
                    parameters.append(coefficients)
                print("Frame {} ".format(frame_Counter))
                print("-"*20)
                for l, par in enumerate(parameters):
                    print("line {} - Coefficients a:{:.4f} b:{:.4f} c:{:.4f}".format(l, par[0], par[1], par[2]))
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
    parser.add_argument('--cuda', type=boolean_string, default=True)
    parser.add_argument('--float16', type=boolean_string, default=True, help="Use float16 for faster inference")
    parser.add_argument('--speed_test', type=boolean_string, default=False,
                        help='Measure inference latency')
    args = parser.parse_args()
    main(args)
    