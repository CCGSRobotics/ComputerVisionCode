#!/usr/bin/env python

#Note this code is a modified version of the ultralytic code found here:
#https://github.com/ultralytics/yolov3

from flask import Flask, render_template, Response, request
import argparse
import time
from sys import platform
import os
from colorama import Fore, Back, Style

from models import *
from utils.datasets import *
from utils.utils import *
from pyzbar.pyzbar import decode

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    while True:
        with torch.no_grad():
            cfg = 'cfg/yolov3.cfg'
            cfghaz = 'cfg/yolov3-tiny-custom.cfg'
            data_cfg = 'directories.data'
            data_cfghaz = 'directories.data'
            weights = os.path.join(os.getcwd(),os.path.join("weights","yolov3.pt"))
            weightshaz = os.path.join(os.getcwd(),os.path.join("weights","bestv1.pt"))
            images = 'data/samples'
            output='output'  # output folder
            img_size=416 #416
            conf_thres=0.02
            nms_thres=0.02
            save_txt=False
            save_images=False
            webcam=True

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

            # Lucas kanade params
            lk_params = dict(winSize = (15, 15),
                             maxLevel = 4,
                             criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) #0.03
            count = 0
            global point, point_selected, old_points
            point_selected = False
            point = ()
            old_points = np.array([[]])
            
            #for i, (path, img, im0, vid_cap) in enumerate(dataloader):
            for i, (path, img, im0) in enumerate(dataloader):
                

                
                im0save = im0.copy()
                count += 1

                if count == 1:
                    start = time.time()
                    height, width = im0.shape[:2]

                if count != 1:
                    p1x_old = p1x
                    p2y_old = p2y
                f = open("output.txt", "r")
                for i in f:
                   haz = i.split()[0]
                   coco = i.split()[1]
                   zbar = i.split()[2]
                   try:
                       p1x = int(float(i.split()[3]) * width)
                       p2y = int(float(i.split()[4]) * height)
                   except:
                       p1x = int(0.5 * width)
                       p2y = int(0.5 * height)
                   p = i.split()[5]
                f.close()
                
                #print(width, height)
                #print(haz, coco)

                t = time.time()
                save_path = str(Path(output) / Path(path).name)

                # Get detections
                img = torch.from_numpy(img).unsqueeze(0).to(device)

                if ONNX_EXPORT:
                    torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
                    return

                
                def select_point(event, x, y, flags, params):
                    global point, point_selected, old_points
                    #print(event, x, y, flags, params)
                    if event == cv2.EVENT_LBUTTONDOWN:
                        point = (x, y)
                        point_selected = True
                        old_points = np.array([[x, y]], dtype=np.float32)

                
                if (point_selected == True):
                    #print("selected")
                    gray_frame = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)

                    cv2.circle(im0, point, 5, (0, 0, 255), 2)
                    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
                    old_gray = gray_frame.copy()
                    old_points = new_points
                    x, y = new_points.ravel()
                    cv2.circle(im0, (x, y), 5, (0, 255, 0), -1)

                if p == "false":
                    point_selected = False

                new = None
                if count != 1:
                    if p1x != p1x_old or p2y != p2y_old:
                        new = True
                    
                if (point_selected == False):
                    if p == "true":
                        old_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY) #im0 is BGR img is RGB
                        select_point(1, p1x, p2y, 1, None)
                else:
                    
                    if new == True:
                        select_point(1, p1x, p2y, 1, None)
                        new = False
                    #print("run")
                    old_gray = gray_frame.copy()
                    
                if haz == "true":
                    predhaz = hazmatmodel(img)
                    detectionshaz = non_max_suppression(predhaz, conf_thres, nms_thres)[0]

                    if detectionshaz is not None and len(detectionshaz) > 0:
                        # Rescale boxes from 416 to true image size
                        scale_coords(img_size, detectionshaz[:, :4], im0.shape).round()

                        # Print results to screen
                        for c in detectionshaz[:, -1].unique():
                            n = (detectionshaz[:, -1] == c).sum()
#-------------------------                            print('%g %ss' % (n, classeshaz[int(c)]), end=', ')

                        # Draw bounding boxes and labels of detections
                        for *xyxy, conf, cls_conf, cls in detectionshaz:

                            # Add bbox to the image
                            label = '%s %.2f' % (classeshaz[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])


                if coco == "true":
                    pred = model(img)
                    detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

                    if detections is not None and len(detections) > 0:
                        # Rescale boxes from 416 to true image size
                        scale_coords(img_size, detections[:, :4], im0.shape).round()

                        # Print results to screen
                        for c in detections[:, -1].unique():
                            n = (detections[:, -1] == c).sum()
#-------------------------                            print('%g %ss' % (n, classes[int(c)]), end=', ')

                        # Draw bounding boxes and labels of detections
                        for *xyxy, conf, cls_conf, cls in detections:

                            # Add bbox to the image
                            label = '%s %.2f' % (classes[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                if zbar == "true":
#-------------------------                    print("zbar")
                    barcodes = decode(im0save)
                    for barcode in barcodes:
                        # extract the bounding box location of the barcode and draw the
                        # bounding box surrounding the barcode on the image
                        (x, y, w, h) = barcode.rect
                        cv2.rectangle(im0, (x, y), (x + w, y + h), (0, 0, 255), 2)

                        # the barcode data is a bytes object so if we want to draw it on
                        # our output image we need to convert it to a string first
                        barcodeData = barcode.data.decode("utf-8")
                        barcodeType = barcode.type

                        # draw the barcode data and barcode type on the image
                        text = "{} ({})".format(barcodeData, barcodeType)
                        cv2.putText(im0, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

                        # print the barcode type and data to the terminal
                        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))

                end = time.time()
                seconds = end - start
                start = time.time()
                if seconds == 0:
                    seconds = 1
                
                cv2.putText(im0, str(round(1/seconds)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)
                
                #cv2.imshow(weights, im0)
                cv2.imwrite('t.jpg', im0)



                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')

                #print(time.time() - t)

                '''if (time.time() - t) > 0.01:
                    print(Fore.RED + 'Done. (%.3fs)' % (time.time() - t))
                else:
                    print(Style.RESET_ALL + 'Done. (%.3fs)' % (time.time() - t))
                '''




            #rval, frame = vc.read()
            #cv2.imwrite('t.jpg', frame)



@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, threaded=True)
