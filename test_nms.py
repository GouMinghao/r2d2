import torch
import torch.nn as nn
import numpy as np
import cv2


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = repeatability == self.max_filter(repeatability)

        # remove low peaks
        maxima *= repeatability >= self.rep_thr
        maxima *= reliability >= self.rel_thr
        return maxima.nonzero().t()[1:3]


if __name__ == "__main__":
    rel = np.fromfile("output/rel.bin", dtype=np.float32).reshape((1, 1, 512, 512))
    rep = np.fromfile("output/rep.bin", dtype=np.float32).reshape((1, 1, 512, 512))
    rel_tensor = torch.from_numpy(rel)
    rep_tensor = torch.from_numpy(rep)
    nms = NonMaxSuppression(0.3, 0.3)
    out = nms(rel_tensor, rep_tensor)

    img = cv2.imread("imgs/boat.png")

    for pt in np.array(out.T):
        print((pt[1], pt[0]))
        img = cv2.circle(img, (pt[1], pt[0]), 1, (255, 0, 0), 1)
    cv2.imwrite("pts.jpg", img)

    print(out)
