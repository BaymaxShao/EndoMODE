import numpy as np
import cv2


img1 = cv2.imread('/home/slj/EndoTraj/NEPose-main/data/chenkun/chenkun_01/lower/2.jpg')
img2 = cv2.imread('/home/slj/EndoTraj/NEPose-main/data/chenkun/chenkun_01/lower/4.jpg')
mask = np.zeros_like(img1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
flow_xy = TVL1.calc(img1, img2, None)

# assert flow_xy.dtype == np.float32
#
# flow_xy = (flow_xy + 15) * (255.0 / (2 * 15))
# flow_xy = np.round(flow_xy).astype(int)
# flow_xy[flow_xy >= 255] = 255
# flow_xy[flow_xy <= 0] = 0

mag, ang = cv2.cartToPolar(flow_xy[:, :, 0], flow_xy[:, :, 1])
mask[:, :, 0] = ang * 180 / np.pi / 2  # 角度
mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
flow = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

cv2.imshow('flow', flow)
cv2.waitKey()
cv2.destroyAllWindows()
