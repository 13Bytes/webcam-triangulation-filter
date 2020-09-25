''' Creates a "triangulated" Picture of you
'''
import sys
import time
import cv2 as cv
import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay

faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Settings
POINT_DENSITY = 11
SENSITIVITY = 10
POINT_DENSITY_FACE = 12
SENSITIVITY_FACE = 4


def is_in_face(faces, x, y):
    for (face_x, face_y, face_w, face_h) in faces:
        if face_x <= x <= face_x + face_w:
            if face_y <= y <= face_y + face_h:
                return True
    return False


if __name__ == "__main__":

    # Default Webcam setzen, Auflösung auslesen
    webcam = cv.VideoCapture(0)
    width = int(webcam.get(3))
    height = int(webcam.get(4))

    if height > 0 and width > 0:
        print('Webcam with resolution of {} x {} detected'.format(width, height))
    else:
        sys.exit('No webcam was found!')

    while True:
        # grep the current from the live-feed
        # frame of type: pixel[y, x] = (r, g, b)
        _, frame = webcam.read()

        # adds a slight blur to reduce flickering in the final feed
        frame = cv.GaussianBlur(frame, (7, 7), 0)
        src_copy = frame.copy()
        src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # laplace corner detection
        ddepth = cv.CV_16S
        kernel_size = 3
        dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
        corner_picture = cv.convertScaleAbs(dst)

        # face detection (for more detailed triangualation)
        faces = faceCascade.detectMultiScale(
            src_gray,
            scaleFactor=1.1,
            minNeighbors=5)

        points = []
        for y in range(1, height, POINT_DENSITY):
            for x in range(1, width, POINT_DENSITY):
                if not is_in_face(faces, x, y):
                    # background-triangulation
                    if (corner_picture[y, x] > SENSITIVITY):
                        points.append([x, y])

        for (face_x, face_y, face_w, face_h) in faces:
            for y in range(face_y, face_y + face_h, POINT_DENSITY_FACE):
                for x in range(face_x, face_x + face_w, POINT_DENSITY_FACE):
                    # face-triangulation
                    pixel = frame[y, x]
                    leftPixel = frame[y, x-1]
                    topPixel = frame[y-1, x]
                    # check if one color changes more than SENSITIVITY_FACE from one pixel to another
                    for i in range(3):
                        # first item is converted to int, so no overflow happens in cv2 datattype
                        dif_a = int(pixel[i]) - leftPixel[i]
                        dif_b = int(pixel[i]) - topPixel[i]

                        if (abs(dif_a) >= SENSITIVITY_FACE) or (abs(dif_b) >= SENSITIVITY_FACE):
                            points.append([x, y])
                            break

        # adds the four cornes of the picture
        points.extend(([0, 0], [0, height], [width, 0], [width, height]))

        # calculate triangles
        points = np.array(points)
        triangles = Delaunay(points)
        triangle_points = points[triangles.simplices]

        for points in triangle_points:
            center_x = int((points[0][0] + points[1][0] + points[2][0]) / 3)
            center_y = int((points[0][1] + points[1][1] + points[2][1]) / 3)
            pixel = frame[center_y, center_x]
            centroid_color = tuple([int(x) for x in pixel])

            src_copy = cv.drawContours(
                src_copy, [points], 0, centroid_color, -1)


        cv.imshow('Video', src_copy)

        # abort with "q"
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

            # to save some CPU performance
        time.sleep(0.03)

    # release the webcam
    webcam.release()
    cv.destroyAllWindows()
