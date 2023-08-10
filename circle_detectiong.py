import cv2
import numpy as np
import glob
import os

# Mendapatkan daftar file gambar dalam direktori
image_folder = './save/OK/*.png'
image_paths = glob.glob(image_folder)

# Loop melalui setiap gambar
for image_path in image_paths:
    # Baca gambar
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Ubah gambar menjadi grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi lingkaran menggunakan transformasi Hough Circle
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=40, param2=20, minRadius=80, maxRadius=90
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Pilih hanya satu lingkaran (misalnya lingkaran pertama)
        circle = circles[0][0]  # Ambil lingkaran pertama

        center = (circle[0], circle[1])
        radius = circle[2]

        # Hitung koordinat bounding box
        x = max(center[0] - radius, 0)
        y = max(center[1] - radius, 0)
        width = min(radius * 2, image.shape[1] - x)
        height = min(radius * 2, image.shape[0] - y)

        # Gambar lingkaran dan bounding box pada gambar asli
        # cv2.circle(image, center, radius, (0, 255, 0), 4)
        # cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Gambar bounding box

        # Cetak koordinat bounding box
        print("Gambar:", image_path)
        print("Koordinat bounding box: X =", x, ", Y =", y, ", Width =", width, ", Height =", height)

        # Tampilkan gambar dengan lingkaran dan bounding box
        cv2.imshow(image_path, image)
        
        # Crop gambar berdasarkan bounding box
        try:       
            cropped_image = image[y:y+height, x:x+width]
            cv2.imwrite("./save/CROP-OK/"+f"{os.path.basename(image_path)}", cropped_image)

            # Tampilkan gambar hasil cropping
            cv2.imshow("Cropped Image", cropped_image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    else:
        print("Tidak ditemukan lingkaran pada gambar:", image_path)
