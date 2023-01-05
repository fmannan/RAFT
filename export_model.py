import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from torch2trt import torch2trt
import tensorrt as trt

from raft.raft import RAFT
from raft.utils import flow_viz
from raft.utils.utils import InputPadder


DEVICE = 'cuda'


def export_trt_model(model, imgL, imgR, fp16, int8, testres, model_filename, outdir):
    model_trt = torch2trt(model, inputs=[imgL, imgR],
                          input_names=["imgL", "imgR"],
                          output_names=["flow"],
                          log_level=trt.Logger.ERROR,
                          fp16_mode=fp16,
                          int8_mode=int8,)
    print('model_trt', model_trt)
    print('model_trt.engine', model_trt.engine)
    extra_suffix = '_fp16' if fp16 else ''
    extra_suffix = '_int8' if int8 else extra_suffix
    extra_suffix += '_resize' if args.resize else ''
    torch.save(model_trt.state_dict(), f'{outdir}/{model_filename}-{testres:0.1f}{extra_suffix}_trt.pth')
    print(f'model stored as {outdir}/{model_filename}-{testres:0.1f}{extra_suffix}_trt.pth')
    with open(f'{outdir}/{model_filename}-{testres:0.1f}{extra_suffix}_trt.engine', 'wb') as f:
        f.write(model_trt.engine.serialize())
    return model_trt


def export(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    image1 = torch.from_numpy(np.random.randn(1, 3, args.height, args.width) * 2 + np.random.rand(1)).float().cuda()
    image2 = torch.from_numpy(np.random.randn(1, 3, args.height, args.width) * 2 + np.random.rand(1)).float().cuda()
    with torch.no_grad():
        print(image1.shape)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_up = model(image1, image2, iters=20, export_mode=True)
        export_trt_model(model, image1, image2, True, False, 1.0, "tmp", args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--outdir', help="Output folder")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--height', type=int, default=436,
                        help='image height. Here we consider the height of the cropped image. \
                              It should be consistent with height_crop in inference time. (default is 855)')
    parser.add_argument('--width', type=int, default=1024,
                        help='image width')
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    export(args)
