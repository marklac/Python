#import numpy as np
import cv2
import dlib


cascPath = "haarcascade_frontalface_default.xml"

# Tworzy kaskadę
faceCascade = cv2.CascadeClassifier(cascPath)

# Szuka
d1 = dlib.get_frontal_face_detector()
d2 = dlib.get_frontal_face_detector()

# Przewiduje
p1 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
p2 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Wczytuje podane zdjęcia
img1 = cv2.imread('obraz7.jpg')
img2 = cv2.imread('obraz9.jpg')

# Tranformuje kolorowe zdjęcie w zdjęcie w odcieniach szaroci
szary1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
szary2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Wyszukuje twarz
twarz1 = faceCascade.detectMultiScale(
    szary1,
    scaleFactor=1.25,
    minNeighbors=5,
    minSize=(30, 30)
)

twarz2 = faceCascade.detectMultiScale(
    szary2,
    scaleFactor=1.25,
    minNeighbors=5,
    minSize=(30, 30)
)

# Rysuje prostokąt w którym znajduje się twarz
for (x, y, w, h) in twarz1:
    cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 5)

for (x, y, w, h) in twarz2:
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 5)
    
# Zaznacza na twarzy punkty z mapy
lica1 = d1(szary1)
for lico1 in lica1:
    x1 = lico1.left() 
    y1 = lico1.top() 
    x2 = lico1.right() 
    y2 = lico1.bottom()
    
    lan1 = p1(image=szary1, box=lico1)
    
    for n in range(0, 68):
        x = lan1.part(n).x
        y = lan1.part(n).y
        
        cv2.circle(img=img1, center=(x, y), radius=3, color=(0, 0, 255), thickness=5)

lica2 = d2(szary2)
for lico2 in lica2:
    x1 = lico2.left() 
    y1 = lico2.top() 
    x2 = lico2.right() 
    y2 = lico2.bottom()
    
    lan2 = p2(image=szary2, box=lico2)
    
    for m in range(0, 68):
        x = lan2.part(m).x
        y = lan2.part(m).y
        
        cv2.circle(img=img2, center=(x, y), radius=3, color=(0, 0, 255), thickness=5)


# Moduł za pomocą którego można pomniejszyć lub powiększyć obraz   
skala1 = 30
width = int(img1.shape[1]*skala1/100)
height = int(img1.shape[0]*skala1/100)
dim1 = (width, height)
skalowany1 = cv2.resize(img1, dim1, interpolation = cv2.INTER_AREA)

skala2 = 30
width = int(img2.shape[1]*skala2/100)
height = int(img2.shape[0]*skala2/100)
dim2 = (width, height)
skalowany2 = cv2.resize(img2, dim2, interpolation = cv2.INTER_AREA)

#Wyświetla wybrane zdjęcia z zaznaczonymi twarzami i dopasowanymi punktami 

okno1 = "Twarz1"
cv2.namedWindow(okno1)
cv2.moveWindow(okno1, 50, 50)
cv2.imshow(okno1, skalowany1)
cv2.waitKey(0)

okno2 = "Twarz2"
cv2.namedWindow(okno2)
cv2.moveWindow(okno2, 50, 50)
cv2.imshow(okno2, skalowany2)
cv2.waitKey(0)


#Sprawdza zgodność twarzy

test_1_img1 = ((lan1.part(45).x-lan1.part(36).x)**2 + (lan1.part(45).y-lan1.part(36).y)**2)**(1/2)
test_1_img2 = ((lan2.part(45).x-lan2.part(36).x)**2 + (lan2.part(45).y-lan2.part(36).y)**2)**(1/2)
test_2_img1 = ((lan1.part(16).x-lan1.part(0).x)**2 + (lan1.part(16).y-lan1.part(0).y)**2)**(1/2)
test_2_img2 = ((lan2.part(16).x-lan2.part(0).x)**2 + (lan2.part(16).y-lan2.part(0).y)**2)**(1/2)
test_3_img1 = ((lan1.part(27).x-lan1.part(33).x)**2 + (lan1.part(27).y-lan1.part(33).y)**2)**(1/2)
test_3_img2 = ((lan2.part(27).x-lan2.part(33).x)**2 + (lan2.part(27).y-lan2.part(33).y)**2)**(1/2)
test_4_img1 = ((lan1.part(35).x-lan1.part(31).x)**2 + (lan1.part(35).y-lan1.part(31).y)**2)**(1/2)
test_4_img2 = ((lan2.part(35).x-lan2.part(31).x)**2 + (lan2.part(35).y-lan2.part(31).y)**2)**(1/2)
test_5_img1 = ((lan1.part(54).x-lan1.part(48).x)**2 + (lan1.part(54).y-lan1.part(48).y)**2)**(1/2)
test_5_img2 = ((lan2.part(54).x-lan2.part(48).x)**2 + (lan2.part(54).y-lan2.part(48).y)**2)**(1/2)
test_6_img1 = ((lan1.part(22).x-lan1.part(21).x)**2 + (lan1.part(22).y-lan1.part(21).y)**2)**(1/2)
test_6_img2 = ((lan2.part(22).x-lan2.part(21).x)**2 + (lan2.part(22).y-lan2.part(21).y)**2)**(1/2)
test_7_img1 = ((lan1.part(26).x-lan1.part(17).x)**2 + (lan1.part(26).y-lan1.part(17).y)**2)**(1/2)
test_7_img2 = ((lan2.part(26).x-lan2.part(17).x)**2 + (lan2.part(26).y-lan2.part(17).y)**2)**(1/2)
test_8_img1 = ((lan1.part(10).x-lan1.part(0).x)**2 + (lan1.part(10).y-lan1.part(0).y)**2)**(1/2)
test_8_img2 = ((lan2.part(10).x-lan2.part(0).x)**2 + (lan2.part(10).y-lan2.part(0).y)**2)**(1/2)
test_9_img1 = ((lan1.part(6).x-lan1.part(16).x)**2 + (lan1.part(6).y-lan1.part(16).y)**2)**(1/2)
test_9_img2 = ((lan2.part(6).x-lan2.part(16).x)**2 + (lan2.part(6).y-lan2.part(16).y)**2)**(1/2)
test_10_img1 = ((lan1.part(57).x-lan1.part(50).x)**2 + (lan1.part(57).y-lan1.part(50).y)**2)**(1/2)
test_10_img2 = ((lan2.part(57).x-lan2.part(50).x)**2 + (lan2.part(57).y-lan2.part(50).y)**2)**(1/2)

g = 0
k = 0

if 0.9*(test_1_img1/test_7_img1) < test_1_img2/test_7_img2 < 1.1*(test_1_img1/test_7_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1
    
if 0.9*(test_1_img1/test_3_img1) < test_1_img2/test_3_img2 < 1.1*(test_1_img1/test_3_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1

if 0.9*(test_3_img1/test_4_img1) < test_3_img2/test_4_img2 < 1.1*(test_3_img1/test_4_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1

if 0.9*(test_1_img1/test_4_img1) < test_1_img2/test_4_img2 < 1.1*(test_1_img1/test_4_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1

if 0.9*(test_5_img1/test_4_img1) < test_5_img2/test_4_img2 < 1.1*(test_5_img1/test_4_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1


if 0.9*(test_6_img1/test_7_img1) < test_6_img2/test_7_img2 < 1.1*(test_6_img1/test_7_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1

if 0.9*(test_6_img1/test_5_img1) < test_6_img2/test_5_img2 < 1.1*(test_6_img1/test_5_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1

if 0.9*(test_10_img1/test_5_img1) < test_10_img2/test_5_img2 < 1.1*(test_10_img1/test_5_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1

if 0.9*(test_3_img1/test_5_img1) < test_3_img2/test_5_img2 < 1.1*(test_3_img1/test_5_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1

if 0.9*(test_4_img1/test_10_img1) < test_4_img2/test_10_img2 < 1.1*(test_4_img1/test_10_img1):
    g = g + 1
    k = k + 1
else:
    g = g
    k = k + 1
    


print("Prawdopodbieństwo, że na obrazach są takie same twarze wynosi", round(g/k*100,2),"%")

cv2.waitKey(0)









