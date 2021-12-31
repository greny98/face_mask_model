import cv2

img = cv2.imread('data/medical_mask/images/5012.jpg')
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
