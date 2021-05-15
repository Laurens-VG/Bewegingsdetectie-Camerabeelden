import cv2
import time
import multiprocessing
from difference_method import diff_color, diff_remove_ghost, diff_color_3
from combined_method import combined_method, combined_method_3
from Edge_based_method import edge_based_method, edge_based_method_3
import numpy as np
import glob
import inspect
from multiprocessing import Pool


def testbench(function, datasetname, stepsize=5, makeVideo=False, show=False, delay=0, start=0):
    """

    :param function: function: de te gebruiken methode
    :param datasetname: string: de foldername van de dataset in de Images map
    :param stepsize: int: het aantal frames tussen de verwerkte frames
    :param makeVideo: boolean: sla de visualizatie op als een testbench.avi
    :param show: boolean: toon de visualizatie van het resultaat
    :param delay: int: als 0 wacht de testbench op een keypress na een frame te tonen, als 1 wacht de testbench niet
    :param start: int: de testbench wordt uitgevoerd vanaf index start
    :return: resultaten: F1-score, precision, recall, fps
    """

    # lees de bestandsnamen
    inNames = glob.glob('Images/' + datasetname + '/input/*.jpg')
    inNames.sort()
    gtNames = glob.glob('Images/' + datasetname + '/groundtruth/*.png')
    gtNames.sort()

    # sla de frames zonder grondwaarheid over
    while True:
        truth = cv2.imread(gtNames[start], cv2.IMREAD_GRAYSCALE)
        if np.all(truth == 85):
            start += 1
        else:
            break

    # initializatie
    tpc = 0
    tnc = 0
    fpc = 0
    fnc = 0
    if makeVideo:
        h, w = cv2.imread(gtNames[0], cv2.IMREAD_GRAYSCALE).shape
        outVideo = cv2.VideoWriter('testbench.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (3 * w, 2 * h))
    if show:
        cv2.namedWindow("testbench", cv2.WINDOW_NORMAL)
    p = Pool(multiprocessing.cpu_count())
    t1 = time.time()

    # controleer of function 2 of 3 frames gebruikt en pas de juiste process_frame toe
    # pas de methode toe via process of process_3 via workers
    # haal de resultaten op en breng de waarden samen
    numparams = len(inspect.signature(function).parameters)
    if numparams == 2:
        idx = range(start, len(inNames))
        res = [None] * len(idx)
        for i in idx:
            in1 = inNames[i - stepsize]
            in2 = inNames[i]
            gt = gtNames[i]
            res[i - start] = p.apply_async(process, (in1, in2, gt, function, makeVideo or show))

        for i in idx:
            frame, result = res[i - start].get()
            tpc += result[0]
            tnc += result[1]
            fpc += result[2]
            fnc += result[3]
            if show:
                cv2.imshow("testbench", frame)
                cv2.waitKey(delay)
            if makeVideo:
                outVideo.write(frame)

    elif numparams == 3:
        idx = range(start, len(inNames) - stepsize)
        res = [None] * len(idx)
        for i in idx:
            in1 = inNames[i - stepsize]
            in2 = inNames[i]
            im3 = inNames[i + stepsize]
            gt = gtNames[i]
            res[i - start] = p.apply_async(process_3, (in1, in2, im3, gt, function, makeVideo or show))

        for i in idx:
            frame, result = res[i - start].get()
            tpc += result[0]
            tnc += result[1]
            fpc += result[2]
            fnc += result[3]
            if show:
                cv2.imshow("testbench", frame)
                cv2.waitKey(delay)
            if makeVideo:
                outVideo.write(frame)
    else:
        raise Exception("expecting function with 2 or 3 parameters")
        # if numparams == 2:
    t2 = time.time()
    p.close()
    p.join()

    if makeVideo:
        outVideo.release()
    if show:
        cv2.destroyAllWindows()

    # bereken de scores
    recall = tpc / (tpc + fnc)
    specficity = tnc / (tnc + fpc)
    fpr = fpc / (fpc + tnc)
    fnr = tnc / (tpc + fnc)
    pwc = 100.0 * (fnc + fpc) / (tpc + fpc + fnc + tnc)
    precision = tpc / (tpc + fpc)
    fmeasure = 2.0 * (recall * precision) / (recall + precision)
    fps = len(gtNames) / (t2 - t1)

    print("time:", t2 - t1)
    print("fps", fps)
    print("recall", recall)
    print("specficity", specficity)
    print("fpc", fpr)
    print("fnc", fnr)
    print("pwc", pwc)
    print("precision", precision)
    print("fmeasure", fmeasure)

    return fmeasure, precision, recall, fps


def process(in1, in2, gt, function, returnViz=False):
    """
    Pas de methode function toe op twee frames in1 en in2 en vergelijk met de grondwaarheid gt
    :param in1: frame 1
    :param in2: frame 2
    :param gt: grondwaarheid van frame 2
    :param function: methode voor segmentatie op basis van drie frames
    :param returnViz: boolean: als True: maak een visualizatie van de resultaten
    :return: [visualizatie, [true positives, true negatives, false positives, false negatives]]
    """
    truth = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
    frame1 = cv2.imread(in1)
    frame2 = cv2.imread(in2)
    segmented = function(frame1, frame2)

    tp = np.logical_and(truth == 255, segmented == 255)
    tn = np.logical_and(truth == 0, segmented == 0)
    fp = np.logical_and(truth == 0, segmented == 255)
    fn = np.logical_and(truth == 255, segmented == 0)

    tpc = np.count_nonzero(tp)
    tnc = np.count_nonzero(tn)
    fpc = np.count_nonzero(fp)
    fnc = np.count_nonzero(fn)

    if not returnViz:
        return segmented, [tpc, tnc, fpc, fnc]
    else:
        viz = np.zeros((truth.shape[0], truth.shape[1], 3), dtype=np.uint8)
        viz[tp] = [255, 255, 0]
        viz[tn] = [255, 0, 0]
        viz[fp] = [255, 0, 255]
        viz[fn] = [0, 0, 255]
        overlay = frame2.copy()
        overlay[segmented == 255] = [0, 0, 255]
        overlay = cv2.addWeighted(frame2, 0.5, overlay, 0.5, 0)
        return np.hstack((overlay, viz)), [tpc, tnc, fpc, fnc]


def process_3(in1, in2, in3, gt, function, returnViz=False):
    """

    :param in1: frame 1
    :param in2: frame 2
    :param in3: frame 3
    :param gt: grondwaarheid van frame 2
    :param function: methode voor segmentatie op basis van drie frames
    :param returnViz: boolean: als True: maak een visualizatie van de resultaten
    :return: [visualizatie, [true positives, true negatives, false positives, false negatives]]
    """
    truth = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
    frame1 = cv2.imread(in1)
    frame2 = cv2.imread(in2)
    frame3 = cv2.imread(in3)
    segmented = function(frame1, frame2, frame3)

    tp = np.logical_and(truth == 255, segmented == 255)
    tn = np.logical_and(truth == 0, segmented == 0)
    fp = np.logical_and(truth == 0, segmented == 255)
    fn = np.logical_and(truth == 255, segmented == 0)

    tpc = np.count_nonzero(tp)
    tnc = np.count_nonzero(tn)
    fpc = np.count_nonzero(fp)
    fnc = np.count_nonzero(fn)

    if not returnViz:
        return None, [tpc, tnc, fpc, fnc]
    else:
        viz = np.zeros((truth.shape[0], truth.shape[1], 3), dtype=np.uint8)
        viz[tp] = [255, 255, 0]
        viz[tn] = [255, 0, 0]
        viz[fp] = [255, 0, 255]
        viz[fn] = [0, 0, 255]
        overlay = frame2.copy()
        overlay[segmented == 255] = [0, 0, 255]
        overlay = cv2.addWeighted(frame2, 0.5, overlay, 0.5, 0)
        return np.hstack((overlay, viz)), [tpc, tnc, fpc, fnc]


if __name__ == '__main__':
    testbench(function=combined_method_3, datasetname="pedestrians", stepsize=5, show=True, makeVideo=False, delay=1)
