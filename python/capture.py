# import cv2

# cam = cv2.VideoCapture(0)
#
# cv2.namedWindow("test")
#
# img_counter = 0
#
# while True:
#     ret, frame = cam.read()
#     cv2.imshow("test", frame)
#     if not ret:
#         break
#     k = cv2.waitKey(1)
#
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         # SPACE pressed
#         img_name = "opencv_frame_{}.png".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         img_counter += 1
#
# cam.release()
#
# cv2.destroyAllWindows()
# Creating database
# It captures images and stores them in datasets
# folder under the folder name of sub_data
import cv2, sys, numpy, os

haar_file = 'python/haarcascade_frontalface_default.xml' #haarcascade_frontalface_alt2.xml'#'

# All the faces data will be
#  present this folder
datasets = 'train_data'

# These are sub data sets of folder,
# for my faces I've used my name you can
# change the label here
sub_data = sys.argv[1]

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

# defining the size of images
(width, height) = (128, 128)

# '0' is used for my webcam,
# if you've any other camera
#  attached use '1' like this
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# The program loops until it has 30 images of the face.
count = 1
while count < 3:
    (ret, im) = webcam.read()
    cv2.imshow("Webcam", im)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        print(len(faces))
        i=0
        for (x, y, w, h) in faces:
            i += 1
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            file_name = '{}/{}_{}_{}.jpg'.format(path, sub_data, count,i)
            print(file_name)
            cv2.imwrite(file_name, face_resize)
        count += 1

webcam.release()
#
cv2.destroyAllWindows()