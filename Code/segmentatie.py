import cv2
import numpy as np
from numba import njit


@njit
def jit_bfs(x, y, padded_image):
    """
    breath first search om het kortste pad te berekenen van x,y tot elk ander punt
    dit is een rekenintensieve lus die met een just in time compiler versneld wordt
    """
    start = (x, y)
    distances = np.full(padded_image.shape, np.inf)  # infinite
    visited = set()
    visited.add(start)
    queue = [start]
    distances[start] = 0
    while queue:
        xv, yv = queue.pop(0)
        neighbours = [(xv - 1, yv), (xv + 1, yv), (xv, yv - 1), (xv, yv + 1)]
        neighbours = [n for n in neighbours if padded_image[n] == 0]
        for neighbour in neighbours:
            if neighbour in visited:
                continue
            distances[neighbour] = distances[xv, yv] + 1
            visited.add(neighbour)
            queue.append(neighbour)

    return distances


def breath_first_search(x, y, image):
    """
    voorbereiding van breath first search om het kortste pad te berekenen van x,y tot elk ander punt
    """
    # voeg padding van 0 en 255 toe aan de afbeelding en bereken de kortste paden
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    padded_image = np.pad(padded_image, pad_width=1, mode='constant', constant_values=255)
    distances = jit_bfs(x + 2, y + 2, padded_image)
    return distances[2:-2, 2:-2]


def path_distance(image):
    """
    kortste pad afstand voor elke pixel tot de vier verschillende hoekpunten
    """
    h, b = image.shape
    d1 = breath_first_search(0, 0, image)
    d2 = breath_first_search(0, b - 1, image)
    d3 = breath_first_search(h - 1, b - 1, image)
    d4 = breath_first_search(h - 1, 0, image)
    res = np.stack((d1, d2, d3, d4))
    return res


def manhattan_distance(image):
    """
    manhattan afstand voor elke pixel tot de vier verschillende hoekpunten
    """
    hoogte, breedte = image.shape
    lijst = []
    lijst.append([np.arange(i, breedte + i) for i in range(hoogte)])
    lijst.append(np.fliplr(lijst[0]))
    lijst.append(np.flipud(lijst[1]))
    lijst.append(np.fliplr(lijst[2]))
    return np.array(lijst)


def excess_distance(image):
    """
    voor elk punt in de afbeelding wordt de excess distance tot de vier hoekpunten berekend
    de excess distance is het verschil tussen de werkelijke afstand en de manhattan afstand tussen twee punten
    voor elke pixel wordt de som genomen van de drie kleinste excess distances
    """
    D = path_distance(image) - manhattan_distance(image)
    D = D.transpose((1, 2, 0))
    D = np.sort(D, axis=2)
    D = D[:, :, 0:3]
    D = np.sum(D, axis=2)
    return D


def segmentatie(image):
    """
    shortest path based interior filling gebaseerd op
    Edge based Foreground Background Estimation with Interior/ExteriorClassification (Gianni Allebosch)
    (https://biblio.ugent.be/publication/5909429/file/5924828.pdf)
    :param image: beeld met bewegende randen
    :return: beeld met bewegende objecten
    """
    result = excess_distance(image) >= 3
    return (255 * result).astype(np.uint8)


if __name__ == "__main__":
    image = cv2.imread("Images/binary2.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("segmentatie", segmentatie(image))
    cv2.imshow("randen", image)
    cv2.waitKey()
