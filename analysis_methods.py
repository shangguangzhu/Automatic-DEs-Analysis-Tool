import sys

sys.path.insert(0, './yolov5')

from yolov5.utils.general import non_max_suppression, scale_coords
import cv2
import torch
import numpy as np
import math


def calculate_coordinate(img, bbox, identities):
    x_img, y_img = img.shape[0], img.shape[1]
    coordinates = np.zeros((int(identities[-1] + 1), 2))
    for t, box in enumerate(bbox):
        id = int(identities[t]) if identities is not None else 0
        x1, y1, x2, y2 = [int(t) for t in box]
        x = ((x2 - x1) / 2 + x1) / x_img
        y = ((y2 - y1) / 2 + y1) / y_img
        coordinates[id, 0] = x
        coordinates[id, 1] = y
    return coordinates


def calculate_diameter_yolo(img, bbox, scale, identities, model2, augment, agnostic_nms):
    radius_outer = np.zeros(int(identities[-1]+1))
    radius_inner = np.zeros(int(identities[-1]+1))
    for i, box_outer in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0
        x1_outer, y1_outer, x2_outer, y2_outer = [int(i) for i in box_outer]
        # Calculate the outer radius of the double emulsion droplet
        dx_outer = x2_outer - x1_outer
        dy_outer = y2_outer - y1_outer
        x_outer = int(dx_outer / 2)
        y_outer = int(dy_outer / 2)
        if dx_outer <= dy_outer:
            r_outer = dx_outer / 2
        else:
            r_outer = dy_outer / 2
        radius_outer[id] = (r_outer * scale)
        # Calculate the inner radius of the double emulsion droplet
        image0 = img[y1_outer:y2_outer, x1_outer:x2_outer]
        image = cv2.resize(image0, (640, 640))
        image = image / 255.
        image = image[:, :, ::-1].transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image.copy())
        image = image.to(torch.float32)
        image = image.cuda()
        # Conduct internal phase detection
        pred_droplets = model2(image, augment=augment)[0]
        pred_droplets.clone().detach()
        pred_droplets = non_max_suppression(
            pred_droplets, 0.5, 0.5, agnostic=agnostic_nms)
        for j, det_droplets in enumerate(pred_droplets):
            if len(det_droplets):
                det_droplets[:, :4] = scale_coords(image.shape[2:], det_droplets[:, :4], image0.shape).round()
                for *box_inner, conf_inner, cls_inner in det_droplets:
                    x1_inner, y1_inner, x2_inner, y2_inner = [int(i) for i in box_inner]
                    dx_inner = x2_inner - x1_inner
                    dy_inner = y2_inner - y1_inner
                    x_inner = int(dx_inner / 2 + x1_inner)
                    y_inner = int(dy_inner / 2 + y1_inner)

                    if dx_inner <= dy_inner:
                        r_inner = int(dx_inner / 2)
                    else:
                        r_inner = int(dy_inner / 2)
                    radius_inner[id] = (r_inner * scale)
            # image1 = cv2.circle(image0, (x_outer, y_outer), int(r_outer), (0, 255, 0), 1)
            # image1 = cv2.circle(image1, (x_inner, y_inner), int(r_inner), (0, 0, 255), 1)
            # cv2.imshow("cropped", image1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    return radius_outer, radius_inner


def calculate_diameter_hough(img, bbox, scale, identities):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_gray = cv2.medianBlur(img_gray, 5)
    radius_outer = np.zeros(int(identities[-1] + 1))
    radius_inner = np.zeros(int(identities[-1] + 1))
    for i, box_outer in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0
        x1_outer, y1_outer, x2_outer, y2_outer = [int(i) for i in box_outer]
        # Calculate the outer radius of the double emulsion droplet
        dx_outer = x2_outer - x1_outer
        dy_outer = y2_outer - y1_outer
        x_outer = int(dx_outer / 2)
        y_outer = int(dy_outer / 2)
        if dx_outer <= dy_outer:
            r_outer = dx_outer / 2
        else:
            r_outer = dy_outer / 2
        radius_outer[id] = (r_outer * scale)
        # Calculate the inner radius of the double emulsion droplet
        image = img[y1_outer:y2_outer, x1_outer:x2_outer]
        image0 = img_gray[y1_outer:y2_outer, x1_outer:x2_outer]
        circles = cv2.HoughCircles(image0, cv2.HOUGH_GRADIENT, 1, 30, param1=180, param2=35, minRadius=0,
                                   maxRadius=0)
        r_inner = 0
        if circles is not None:
            for circle in circles[0]:
                if circle[2] < r_outer and circle[2] != 0:
                    # print(circle[2])
                    x_inner = int(circle[0])
                    y_inner = int(circle[1])
                    if r_inner < circle[2]:
                        r_inner = circle[2]
            radius_inner[id] = (r_inner * scale)
        # image1 = cv2.circle(image, (x_outer, y_outer), int(r_outer), (0, 255, 0), 1)
        # image1 = cv2.circle(image1, (x_inner, y_inner), int(r_inner), (0, 0, 255), 1)
        # cv2.imshow("cropped", image1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return radius_outer, radius_inner


def calculate_diameter_contour(img, bbox, scale, identities):
    radius_outer = np.zeros(int(identities[-1]+1))
    radius_inner = np.zeros(int(identities[-1]+1))
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.threshold(img1, 156, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((1, 1), np.uint8)
    img1 = cv2.dilate(img1, kernel, iterations=4)
    img1 = cv2.erode(img1, kernel, iterations=4)
    # dilation = cv2.dilate(erosion, kernel, iterations=5)
    img1 = cv2.GaussianBlur(img1, (1, 1), 0)


    # img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # img_gray = cv2.medianBlur(img_gray, 5)
    for i, box_outer in enumerate(bbox):
        id = int(identities[i]) if identities is not None else 0
        x1_outer, y1_outer, x2_outer, y2_outer = [int(i) for i in box_outer]
        # Calculate the outer radius of the double emulsion droplet
        dx_outer = x2_outer - x1_outer
        dy_outer = y2_outer - y1_outer
        x_outer = int(dx_outer / 2)
        y_outer = int(dy_outer / 2)
        if dx_outer <= dy_outer:
            r_outer = dx_outer / 2
        else:
            r_outer = dy_outer / 2
        radius_outer[id] = (r_outer * scale)
        # Calculate the inner radius of the double emulsion droplet
        image0 = img1[y1_outer:y2_outer, x1_outer:x2_outer]
        image = img[y1_outer:y2_outer, x1_outer:x2_outer]
        contours, hierarchy = cv2.findContours(image0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        r_inner = 0
        if contours is not None:
            for contour in range(len(contours)):
                area = cv2.contourArea(contours[contour])
                rr = math.sqrt(area / math.pi)
                if rr > 2 and rr < 100 and rr > r_inner:
                    r_inner = rr
            radius_inner[id] = (r_inner * scale)
        # image1 = cv2.circle(image, (x_outer, y_outer), int(r_outer), (0, 255, 0), 1)
        # image1 = cv2.drawContours(image1, contours, -1, (0, 200, 255), 1)
        # cv2.imshow("cropped", image1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return radius_outer, radius_inner
