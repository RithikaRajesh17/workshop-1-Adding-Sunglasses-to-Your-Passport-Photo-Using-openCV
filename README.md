# Workshop--1-Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV

## AIM

To develop a Python program using OpenCV and NumPy to overlay sunglasses onto a person's face using image processing techniques such as image resizing, alpha masking, and arithmetic blending.

## ALGORITHM

1. Import the required libraries: OpenCV, NumPy, and Matplotlib.
2. Read the face image using `cv2.imread()`.
3. Read the sunglasses PNG image with the alpha channel using `cv2.imread('sunglasses.png', -1)`.
4. Display the original face and sunglasses images.
5. Resize the sunglasses image to fit the selected eye region of the face.
6. Separate the sunglasses image into colour channels (BGR) and the alpha channel.
7. Normalize the alpha channel by dividing its pixel values by 255.
8. Merge the alpha channel into three channels to match the face image dimensions.
9. Create a copy of the original face image.
10. Extract the eye region (ROI – Region of Interest) using the appropriate pixel coordinates.
11. Apply the inverse alpha mask to the eye region to remove the pixels covered by the sunglasses.
12. Apply the alpha mask to the sunglasses image to retain its visible portions.
13. Combine the masked eye region and masked sunglasses using arithmetic addition.
14. Replace the original eye region with the blended result.
15. Display the original face image and the final image with sunglasses.

## PROGRAM

```python
Developed by
Name: Rithika R
Reg.No: 212224240136



import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Face Image
faceImage = cv2.imread('img1.jpeg')

plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Original Face")
plt.show()

# Load Sunglasses Image with Alpha Channel
glassPNG = cv2.imread('sunglasses.png', -1)

# Resize sunglasses to fit the eye region
glassPNG = cv2.resize(glassPNG, (350, 150))
print("Image Dimension:", glassPNG.shape)

# Separate Color and Alpha Channels
glassBGR = glassPNG[:, :, 0:3]
glassMask1 = glassPNG[:, :, 3] / 255.0

# Display Color and Alpha Channels
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(cv2.cvtColor(glassBGR, cv2.COLOR_BGR2RGB))
plt.title("Sunglasses Color Channels")

plt.subplot(122)
plt.imshow(glassMask1, cmap='gray')
plt.title("Sunglasses Alpha Channel")

plt.show()

# Merge Alpha Mask into Three Channels
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Create a Copy of Face Image
faceWithGlasses = faceImage.copy()

# Extract Eye Region
eyeROI = faceWithGlasses[325:475, 220:570]

# Convert Images to Float32
eyeROI = eyeROI.astype(np.float32)
glassBGR = glassBGR.astype(np.float32)
glassMask = glassMask.astype(np.float32)

# Apply Mask to Eye Region
maskedEye = cv2.multiply(eyeROI, (1 - glassMask))

# Apply Mask to Sunglasses
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Combine Both Images
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Convert Back to Uint8
eyeRoiFinal = np.uint8(eyeRoiFinal)

# Replace Eye Region
faceWithGlasses[325:475, 220:570] = eyeRoiFinal

# Display Original and Final Images
plt.figure(figsize=(15, 8))

plt.subplot(121)
plt.imshow(cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB))
plt.title("Original Image")

plt.subplot(122)
plt.imshow(cv2.cvtColor(faceWithGlasses, cv2.COLOR_BGR2RGB))
plt.title("Face with Sunglasses")

plt.show()
```

## OUTPUT
Original Image

<img width="334" height="434" alt="download" src="https://github.com/user-attachments/assets/2eab01c7-0dfe-4d9e-bb50-8ecf8be7f735" />


Sunglasses

<img width="552" height="280" alt="download" src="https://github.com/user-attachments/assets/16e5f6be-7170-460e-90b0-2a05c5995b45" />

<img width="1210" height="203" alt="download" src="https://github.com/user-attachments/assets/cec0638e-b8e5-46b7-b41b-50009b78bfdc" />



<img width="334" height="418" alt="download" src="https://github.com/user-attachments/assets/8d5a9414-cbc7-4f79-86d8-a5baa86bf573" />


<img width="1597" height="184" alt="download" src="https://github.com/user-attachments/assets/17ec31b1-80b2-4eca-8e4a-7f801b7dd83e" />



<img width="1606" height="1003" alt="download" src="https://github.com/user-attachments/assets/06939ba6-c549-4eeb-b465-eb423fa54c40" />

## RESULT

The sunglasses were successfully overlaid onto the face image using OpenCV and NumPy. The alpha masking technique was used to preserve the transparency of the sunglasses, while arithmetic blending combined the sunglasses with the selected eye region to produce the final augmented image.
