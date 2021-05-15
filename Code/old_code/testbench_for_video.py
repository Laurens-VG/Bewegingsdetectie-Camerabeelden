import cv2
from difference_method import diff_method, diff_color, diff_remove_ghost
from Edge_based_method import edge_based_method
import os
from glob import glob
import numpy as np
import time
from multiprocessing.pool import ThreadPool
from collections import deque

def combined_method(image1, image2):
    result1 = diff_color(image1, image2)
    result2 = edge_based_method(image1, image2)
    combine = np.bitwise_and(result1, result2)
    cv2.imshow("combined method steps", np.hstack((result1, result2, combine)))
    return result1, result2, combine

class TestBench:
    def __init__(self, datasetname, stepsize=5, show=True, delay=1, start=0, make_video=False):
        self.function = combined_method
        self.stepsize = stepsize
        self.show = show
        self.delay = delay
        self.start = start
        self.make_video = make_video

        self.gtFolder = "Images/" + datasetname + "/groundtruth/"
        self.inFolder = "Images/" + datasetname + "/input/"
        dir = os.getcwd()
        os.chdir(self.inFolder)
        self.inNames = glob('*.jpg')
        self.inNames.sort()
        os.chdir(dir)
        os.chdir(self.gtFolder)
        self.gtNames = glob('*.png')
        self.gtNames.sort()
        os.chdir(dir)

        self.count_time = 0
        self.count_frames = 0
        self.frame_count = self.start
        self.tpc = 0
        self.tnc = 0
        self.fpc = 0
        self.fnc = 0

        self.temp = 0

        if self.show:
            self.window = cv2.namedWindow("Testbench", cv2.WINDOW_KEEPRATIO)
        else:
            self.window = None
        if self.make_video:
            h, w = cv2.imread(self.gtFolder + self.gtNames[0], cv2.IMREAD_GRAYSCALE).shape
            self.out_video = cv2.VideoWriter('testbench.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (3 * w, 2 * h))
        else:
            self.out_video = None

    def run(self):
        while self.frame_count < len(self.gtNames) - self.stepsize:
            frame1 = cv2.imread(self.inFolder + self.inNames[self.frame_count])
            frame2 = cv2.imread(self.inFolder + self.inNames[self.frame_count + self.stepsize])
            truth = cv2.imread(self.gtFolder + self.gtNames[self.frame_count + self.stepsize],
                               flags=cv2.IMREAD_GRAYSCALE)
            self.frame_count += 1
            self.temp += 1

            if np.all(truth == 85):
                # no truth available
                continue

            start = time.time()
            method1, method2, segmented = self.function(frame1, frame2)
            end = time.time()
            self.count_frames += 1
            self.count_time += (end - start)
            self.process_result(frame1, frame2, truth, method1, method2,segmented)

        if self.make_video:
            self.out_video.release()
        if self.show:
            cv2.destroyAllWindows()

        fps = self.count_frames / self.count_time
        recall = self.tpc / (self.tpc + self.fnc)
        specficity = self.tnc / (self.tnc + self.fpc)
        fpr = self.fpc / (self.fpc + self.tnc)
        fnr = self.tnc / (self.tpc + self.fnc)
        pwc = 100.0 * (self.fnc + self.fpc) / (self.tpc + self.fpc + self.fnc + self.tnc)
        precision = self.tpc / (self.tpc + self.fpc)
        fmeasure = 2.0 * (recall * precision) / (recall + precision)

        print("recall", recall)
        print("specficity", specficity)
        print("fpc", fpr)
        print("fnc", fnr)
        print("pwc", pwc)
        print("precision", precision)
        print("fmeasure", fmeasure)
        print("fps", fps)
        return fmeasure, precision, recall, fps

    def process_result(self, frame1, frame2, truth, method1, method2, segmented):
        tp = np.logical_and(truth == 255, segmented == 255)
        tn = np.logical_and(truth == 0, segmented == 0)
        fp = np.logical_and(truth == 0, segmented == 255)
        fn = np.logical_and(truth == 255, segmented == 0)

        self.tpc += np.count_nonzero(tp)
        self.tnc += np.count_nonzero(tn)
        self.fpc += np.count_nonzero(fp)
        self.fnc += np.count_nonzero(fn)

        if self.show or self.make_video:
            viz = np.zeros((truth.shape[0], truth.shape[1], 3), dtype=np.uint8)
            viz[tp] = [255, 255, 0]
            viz[tn] = [255, 0, 0]
            viz[fp] = [255, 0, 255]
            viz[fn] = [0, 0, 255]
            segmented = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
            method1 = cv2.cvtColor(method1, cv2.COLOR_GRAY2BGR)
            method2 = cv2.cvtColor(method2, cv2.COLOR_GRAY2BGR)
            truth = cv2.cvtColor(truth, cv2.COLOR_GRAY2BGR)
            original = np.vstack((frame1, frame2))
            method12 = np.vstack((method1, method2))
            viz = np.vstack((viz, segmented))
            viz = np.hstack((original, method12, viz))
            if self.show:
                cv2.imshow("Testbench", viz)
                cv2.waitKey(self.delay)
            if self.make_video:
                self.out_video.write(viz)


if __name__ == "__main__":
    testbench = TestBench(datasetname="highway", stepsize=5, show=False, delay=0, start=0,
                          make_video=True)
    testbench.run()
