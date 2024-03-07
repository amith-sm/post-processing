import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

img_dir = r"C:\Users\transponster\Documents\nortera\training\train\images"
annotations_dir = r"C:\Users\transponster\Documents\nortera\training\train\labels"
classification_dir = r"D:\anshul\notera\data\class"

for img_ in os.listdir(img_dir):
    if ".jpg" in img_:
        img_p = os.path.join(img_dir, img_)
        label_p = os.path.join(annotations_dir, img_.replace(".jpg", ".txt"))
        img = cv2.imread(img_p)
        dh, dw, _ = img.shape

        fl = open(label_p, 'r')
        data = fl.readlines()
        fl.close()

        counter = 1
        for dt in data:
            cls, x, y, w, h = dt.split(' ')
            nx = int(float(x) * dw)
            ny = int(float(y) * dh)
            nw = int(float(w) * dw)
            nh = int(float(h) * dh)

            x_start = int(nx - (nw/2))
            y_start = int(ny - (nh/2))
            x_end = int(x_start + nw)
            y_end = int(y_start + nh)

            cls_dir = os.path.join(classification_dir, str(cls))
            os.makedirs(cls_dir, exist_ok=True)
            try:
                img_cls = img[y_start: y_end, x_start: x_end, :]

                cv2.imwrite(os.path.join(cls_dir, f"{img_.replace('.jpg', '')}_{counter}.png"), img_cls)
                # plt.imsave(os.path.join(cls_dir, f"{img_.replace('.jpg', '')}_{counter}.png"), img_cls)
            except:
                pass
            # break
            counter += 1
        # break
