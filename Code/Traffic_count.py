import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from difference_method import diff_color, diff_color_3
import collections
from skvideo.io import FFmpegReader
import argparse

# Hoekpunten Gopro
# Links bovenaan
x1 = 650 // 2
y1 = 280 // 2
# Rechts beneden
x2 = 850 // 2
y2 = 380 // 2

# Area 4 GoPro
# Links bovenaan
x3 = 70 // 2
y3 = 370 // 2
# Rechts beneden
x4 = 270 // 2
y4 = 470 // 2

# Counter array
values_1 = []
values_2 = []
count = []
i = 0
threshold = 175

# Counter Real Time
k = 0
vol_1 = 0
vol_2 = 0


def area(segmented_image, x1, y1, x2, y2):
    """
    snij de rechthoek bepaald door twee hoekpunten (x1, y1) en (x2, y2) uit een afbeelding
    """
    cropped_im = segmented_image[y1:y2, x1:x2]
    # cv2.imshow("Area_count", segmented_image)
    # cv2.imshow("Area_cropped", cropped_im)
    # cv2.waitKey()
    return cropped_im


def image_values(cropped_im):
    """
    bereken de pixelwaardesom en oppervlakte van een afbeelding
    """
    count = np.sum(cropped_im)
    length = np.size(cropped_im)
    # print(cropped_im)
    # print("length = ", length)
    # print("som = ", som)
    return count, length


def traffic(image1, image2, image3, x1, y1, x2, y2, x3, y3, x4, y4):
    """
    bereken de segmentatie op basis van de drie frames
    beslist of auto aanwezig is op basis van aantal witte pixels in rechthoek
    :param image1, image2, image3: drie frames
    :param x1, y1, x2, y2: twee hoekpunten van de eerste rechthoek
    :param x3, y3, x4, y4: twee hoekpunten van de tweede rechthoek
    :return: frame met visualizatie
    """
    # Methode_setup
    result = diff_color_3(image1, image2, image3)  # kan een andere methode zijn
    cropped_1 = area(result, x1, y1, x2, y2)
    cropped_2 = area(result, x3, y3, x4, y4)
    count_1, length_1 = image_values(cropped_1)
    count_2, length_2 = image_values(cropped_2)

    overlay = image2.copy()
    overlay[result == 255] = [0, 0, 255]
    overlay = cv2.addWeighted(overlay, 0.5, image2, 0.5, 0)
    # Counter realtime
    show_counter(overlay)
    counter_realtime(count_1, count_2, length_1, length_2)

    # Counter array
    value_plot_1 = count_1 / length_1
    value_plot_2 = count_2 / length_2
    counter_array(value_plot_1, value_plot_2)  # enkel om plot te tonen, kan weggelaten worden

    return overlay


def show_counter(image):
    """
    voegt text met de huidige waarde van de couter en rechthoeken toe aan het frame
    """
    global k
    # Counter tekst
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Counter: ' + str(k), (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # rechthoek
    points_1 = np.array([[x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    points_2 = np.array([[x3, y4], [x4, y4], [x4, y3], [x3, y3]])
    cv2.polylines(image, [points_1], 1, 180)
    cv2.polylines(image, [points_2], 1, 180)


def counter_realtime(value_1, value_2, length_1, length_2):
    """
    verhoog de counter als er meer dan 80% van de rechthoek wit is
    er kan pas verder worden geteld als de rechthoek 5 frames leeg is
    """
    global k
    global vol_1
    global vol_2
    rt_threshold = 0.8
    max_vol = 5

    # print("value_1", value_1)
    if value_1 > length_1 * 255 * rt_threshold:
        if vol_1 == 0:
            k += 1
            vol_1 = max_vol
    else:
        if vol_1 > 0:
            vol_1 -= 1

    # print("value_2: ", value_2)
    if value_2 > length_2 * 255 * rt_threshold:
        if vol_2 == 0:
            k += 1
            vol_2 = max_vol
    else:
        if vol_2 > 0:
            vol_2 -= 1


def counter_array(value_1, value_2):
    """
    voeg de nieuwe waarden toe aan de counter array om de plot van het verloop van het aantal witte pixels te kunnen tonen
    """
    global values_1, values_2, i, count

    values_1 = np.append(values_1, value_1)
    values_2 = np.append(values_2, value_2)
    i += 1
    count = np.append(count, i)


def total_count(array):
    """
    berekening van aantal auto's uit het volledige verloop van het aantal witte pixels
    """
    global threshold
    i = 0
    total = 0

    while i < len(array):
        if array[i] > threshold:
            total += 1
            while i < len(array) and array[i] > threshold:
                i += 1
        i += 1
    return total


# Testbench Datasets
def iterate_all_images(datasetname, x1, y1, x2, y2, x3, y3, x4, y4, stepsize=5):
    """
    itereer over alle frames in een dataset
    tel het aantal wagens die door een van de twee rechthoeken rijden
    de visualizatie wordt opgeslaan als video in trafic.avi
    :param filename: naam van de dataset
    :param x1, y1, x2, y2: twee hoekpunten van de eerste rechthoek
    :param x3, y3, x4, y4: twee hoekpunten van de tweede rechthoek
    :return: None
    """
    inFolder = "Images/" + datasetname + "/input/"
    dir = os.getcwd()
    os.chdir(inFolder)
    inNames = glob('*.jpg')
    inNames.sort()
    os.chdir(dir)
    frame_count = stepsize
    shape = cv2.imread(inFolder + inNames[0]).shape
    video_writer = cv2.VideoWriter('traffic.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (shape[1], shape[0]))
    while frame_count < len(inNames) - stepsize:
        frame0 = cv2.imread(inFolder + inNames[frame_count - stepsize])
        frame1 = cv2.imread(inFolder + inNames[frame_count])
        frame2 = cv2.imread(inFolder + inNames[frame_count + stepsize])
        frame = traffic(frame0, frame1, frame2, x1, y1, x2, y2, x3, y3, x4, y4)
        cv2.imshow("Traffic", frame)
        cv2.waitKey(1)
        frame_count += 1
        video_writer.write(frame)
    video_writer.release()


# Testbench GoPro
def iterate_video(filename, x1, y1, x2, y2, x3, y3, x4, y4, down_scale=True):
    """
    itereer over alle frames van de video
    tel het aantal wagens die door een van de twee rechthoeken rijden
    de visualizatie wordt opgeslaan als video in trafic.avi
    druk Q om te stoppen
    :param filename: bestandsnaam van de video
    :param x1, y1, x2, y2: twee hoekpunten van de eerste rechthoek
    :param x3, y3, x4, y4: twee hoekpunten van de tweede rechthoek
    :param down_scale: boolean: als True wordt de resolutie van de video gehalveerd
    :return: None
    """
    queue = collections.deque()
    if not os.path.isfile(filename):
        raise Exception("file not found")
    reader = FFmpegReader(filename)
    shape = reader.getShape()[1:3]
    if down_scale:
        shape = [shape[0] // 2, shape[1] // 2]
    stepsize = 5
    video_writer = cv2.VideoWriter('traffic.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (shape[1], shape[0]))
    for frame in reader.nextFrame():
        if down_scale:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        queue.append(frame[:, :, ::-1])
        if len(queue) > 2 * stepsize:
            res = traffic(queue[0], queue[stepsize], queue[stepsize * 2], x1, y1, x2, y2, x3, y3, x4, y4)
            cv2.imshow("Traffic", res)
            k = cv2.waitKey(1)
            queue.popleft()
            video_writer.write(res)
            if k == 113:
                # press Q to break
                break
    video_writer.release()


if __name__ == "__main__":
    # # Area 1 Highway
    # # Links bovenaan
    # x1 = 50
    # y1 = 172
    # # Rechts beneden
    # x2 = 120
    # y2 = 188
    #
    # # Area 2 Highway
    # # Links bovenaan
    # x3 = 160
    # y3 = 175
    # # Rechts beneden
    # x4 = 230
    # y4 = 190
    # # Dataset
    # iterate_all_images("highway", x1, y1, x2, y2, x3, y3, x4, y4, 5)

    cv2.namedWindow("Traffic", cv2.WINDOW_NORMAL)

    # Video
    parser = argparse.ArgumentParser(description='Testbench bewegingsdetectie')
    parser.add_argument("--filename", help="filename of video in folder Video", default="GOPR0022.MP4")
    iterate_video("Video/" + parser.parse_args().filename, x1, y1, x2, y2, x3, y3, x4, y4)

    # Plot bij dataset
    print(total_count(values_1))
    print(total_count(values_2))
    plt.plot(count, values_1)
    plt.plot(count, values_2)
    plt.xlabel('Frames')
    plt.ylabel('Value')
    plt.show()
