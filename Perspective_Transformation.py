import cv2
import numpy as np


def video_transform(im):  # 输入需要计算的三个区域的四个角点，将其进行变换并且进行拼接
    list1 = [(68, 12), (42, 50), (318, 272), (349, 235)]
    list2 = [(15, 583), (28, 605), (340, 354), (327, 334)]
    list3 = [(550, 275), (550, 330), (894, 330), (894, 275)]
    list1 = np.float32(list1)
    list2 = np.float32(list2)
    list3 = np.float32(list3)
    p11, p12, p13, p14 = list1
    p21, p22, p23, p24 = list2
    p31, p32, p33, p34 = list3
    h1 = int(max(np.sqrt(np.sum(np.square(p12-p11))), np.sqrt(np.sum(np.square(p13-p14)))))
    w1 = int(max(np.sqrt(np.sum(np.square(p14-p11))), np.sqrt(np.sum(np.square(p13-p12)))))
    h2 = int(max(np.sqrt(np.sum(np.square(p22-p21))), np.sqrt(np.sum(np.square(p23-p24)))))
    w2 = int(max(np.sqrt(np.sum(np.square(p24-p21))), np.sqrt(np.sum(np.square(p23-p22)))))
    h3 = int(max(np.sqrt(np.sum(np.square(p32-p31))), np.sqrt(np.sum(np.square(p33-p34)))))
    w3 = int(max(np.sqrt(np.sum(np.square(p34-p31))), np.sqrt(np.sum(np.square(p33-p32)))))
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]])
    pts3 = np.float32([[0, 0], [0, h3], [w3, h3], [w3, 0]])
    matrix1 = cv2.getPerspectiveTransform(list1, pts1)
    matrix2 = cv2.getPerspectiveTransform(list2, pts2)
    matrix3 = cv2.getPerspectiveTransform(list3, pts3)
    # im = np.float32(im)
    result1 = cv2.warpPerspective(im, matrix1, (w1, h1))
    result2 = cv2.warpPerspective(im, matrix2, (w2, h2))
    result3 = cv2.warpPerspective(im, matrix3, (w3, h3))

    output_image_frame = np.zeros(((h1 + h2 + h3), max(w1, w2, w3),  3))
        # output_image_frame = np.float32(output_image_frame)

    output_image_frame[:h1, :w1, :] = result1.copy()
    output_image_frame[h1:h1+h2, :w2, :] = result2.copy()
    output_image_frame[h1+h2:, :w3, :] = result3.copy()
    output_image_frame = np.array(output_image_frame, dtype=np.uint8)
    return output_image_frame
