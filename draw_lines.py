import cv2
# Load image, convert to grayscale, Otsu's threshold
image = cv2.imread('images/20240110_064610.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# Find number of rows
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
horizontal = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
rows = 0
for c in cnts:
    cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
    rows += 1
# Find number of columns
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 50))
vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                            vertical_kernel, iterations=2)
cnts = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
columns = 0
for c in cnts:
    cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
    columns += 1
print('Rows:', rows - 1)
print('Columns:', columns - 1)
thresh = cv2.resize(thresh, (500, 500))
image = cv2.resize(image, (500, 500))
cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.waitKey()
