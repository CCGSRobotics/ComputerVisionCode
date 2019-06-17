import argparse
import time
from sys import platform
import os
from colorama import Fore, Back, Style 

from models import *
from utils.datasets import *
from utils.utils import *

def detect(
        cfg,
        cfghaz,
        data_cfg,
        data_cfghaz,
        weights,
        weightshaz,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=False,
        webcam=True
):

    

    device = torch_utils.select_device()

    # Initialize model

    model = Darknet(cfg, img_size)
    hazmatmodel = Darknet(cfghaz, img_size)
    

    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    hazmatmodel.load_state_dict(torch.load(weightshaz, map_location=device)['model'])

    model.to(device).eval()
    hazmatmodel.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    #start webcam

    save_images = False
    dataloader = LoadWebcam(img_size=img_size)


    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'], "example_single_class.names")
    classeshaz = load_classes(parse_data_cfg(data_cfghaz)['names'], "hazmat.names")
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    #for i, (path, img, im0, vid_cap) in enumerate(dataloader):
    for i, (path, img, im0) in enumerate(dataloader):
        
        f = open("output.txt", "r")
        for i in f:
           haz = i.split()[0]
           coco = i.split()[1]
        f.close()

        #print(haz, coco)
    
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return


        if haz == "True":
            predhaz = hazmatmodel(img)
            detectionshaz = non_max_suppression(predhaz, conf_thres, nms_thres)[0]

            if detectionshaz is not None and len(detectionshaz) > 0:
                # Rescale boxes from 416 to true image size
                scale_coords(img_size, detectionshaz[:, :4], im0.shape).round()

                # Print results to screen
                for c in detectionshaz[:, -1].unique():
                    n = (detectionshaz[:, -1] == c).sum()
                    print('%g %ss' % (n, classeshaz[int(c)]), end=', ')

                # Draw bounding boxes and labels of detections
                for *xyxy, conf, cls_conf, cls in detectionshaz:

                    # Add bbox to the image
                    label = '%s %.2f' % (classeshaz[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])


        if coco == "True":
            pred = model(img)
            detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

            if detections is not None and len(detections) > 0:
                # Rescale boxes from 416 to true image size
                scale_coords(img_size, detections[:, :4], im0.shape).round()

                # Print results to screen
                for c in detections[:, -1].unique():
                    n = (detections[:, -1] == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')

                # Draw bounding boxes and labels of detections
                for *xyxy, conf, cls_conf, cls in detections:

                    # Add bbox to the image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        cv2.imshow(weights, im0)

        print(time.time() - t)

        if (time.time() - t) > 0.01:
            print(Fore.RED + 'Done. (%.3fs)' % (time.time() - t))
        else:
            print(Style.RESET_ALL + 'Done. (%.3fs)' % (time.time() - t))

        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-custom.cfg', help='cfg file path')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--cfghaz', type=str, default='cfg/yolov3-tiny-custom.cfg', help='cfg file path for hazmats')
    parser.add_argument('--data-cfg', type=str, default='directories.data', help='coco.data file path')
    parser.add_argument('--data-cfghaz', type=str, default='directories.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default=os.path.join(os.getcwd(),os.path.join("weights","yolov3.pt")), help='path to weights file')
    parser.add_argument('--weightshaz', type=str, default=os.path.join(os.getcwd(),os.path.join("weights","best.pt")), help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.2, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.cfghaz,
            opt.data_cfg,
            opt.data_cfghaz,
            opt.weights,
            opt.weightshaz,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )
