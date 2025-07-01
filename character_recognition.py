# import cv2
# import numpy as np
# from keras.models import load_model
# from utils import preprocess_character
#
# # Load pre-trained character recognition model
# model = load_model("D:/ANPR project/code/python/checkpoints/model_epoch_15_.keras")
#
# # Character mapping
# class_names = {i: str(i) for i in range(10)}  # Digits 0-9
# class_names.update({10: 'ba', 11: 'baa', 12: 'bhe', 13: 'c', 14: 'cha', 15: 'di',
#                     16: 'ga', 17: 'ha', 18: 'ja', 19: 'jha', 20: 'ka', 21: 'kha',
#                     22: 'ko', 23: 'lu', 24: 'ma', 25: 'me', 26: 'naa', 27: 'nya',
#                     28: 'pa', 29: 'pra', 30: 'se', 31: 'su', 32: 'ta', 33: 'ya'})
#
# def recognize_plate_characters(plate_img):
#     """ Recognizes characters from a two-row number plate image. """
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian Blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 1)
#
#     # Apply Otsu's Thresholding
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # Apply Morphological Operations (Erosion)
#     eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
#
#     # Find contours of characters
#     contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
#
#     # Filter valid bounding boxes based on size constraints
#     valid_boxes = [(x, y, w, h) for (x, y, w, h) in bounding_boxes if 12 < w < 200 and 17 < h < 200]
#
#     if not valid_boxes:
#         return "No characters detected"
#
#     # Sort bounding boxes by y-coordinate first (row-wise sorting)
#     valid_boxes.sort(key=lambda b: b[1])
#
#     # Determine row separation based on height
#     heights = [y for _, y, _, _ in valid_boxes]
#     median_height = np.median(heights)
#
#     top_row = [box for box in valid_boxes if box[1] < median_height]
#     bottom_row = [box for box in valid_boxes if box[1] >= median_height]
#
#     # Sort characters in each row from left to right
#     top_row.sort(key=lambda b: b[0])
#     bottom_row.sort(key=lambda b: b[0])
#
#     # Recognize characters from sorted bounding boxes
#     plate_text = ""
#
#     for row in [top_row, bottom_row]:
#         for x, y, w, h in row:
#             char_crop = eroded[y:y+h, x:x+w]
#             char_crop = cv2.bitwise_not(char_crop)  # Invert for better recognition
#             char_crop = preprocess_character(char_crop)  # Resize & normalize
#
#             prob = model.predict(char_crop)[0]
#             idx = np.argmax(prob)
#             plate_text += class_names.get(idx, "?")
#
#         plate_text += " "  # Space to separate rows
#
#     return plate_text.strip()
#
# import cv2
# import numpy as np
# from keras.models import load_model
# from utils import preprocess_character
#
# # Load pre-trained character recognition model
# model = load_model("D:/ANPR project/code/python/checkpoints/model_epoch_15_.keras")
#
# # Character mapping
# class_names = {i: str(i) for i in range(10)}  # Digits 0-9
# class_names.update({10: 'ba', 11: 'baa', 12: 'bhe', 13: 'c', 14: 'cha', 15: 'di',
#                     16: 'ja', 17: 'ha', 18: 'ja', 19: 'jha', 20: 'ka', 21: 'kha',
#                     22: 'ko', 23: 'lu', 24: 'ma', 25: 'me', 26: 'naa', 27: 'nya',
#                     28: 'pa', 29: 'pra', 30: 'se', 31: 'su', 32: 'ta', 33: 'ya'})
#
# def recognize_plate_characters(plate_img, debug=False):
#     """Detects and recognizes characters from a license plate image and returns as plain text."""
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian Blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 1)
#
#     # Apply Otsu's Thresholding for binarization
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # Apply Morphological Operations (Erosion) for noise reduction
#     eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
#
#     # Find contours of characters
#     contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
#
#     # Filter valid bounding boxes based on size constraints
#     valid_boxes = [(x, y, w, h) for (x, y, w, h) in bounding_boxes if 12 < w < 200 and 17 < h < 200]
#
#     if not valid_boxes:
#         return "No characters detected"
#
#     # Sort bounding boxes by y-coordinate first (to determine rows)
#     valid_boxes.sort(key=lambda b: b[1])
#
#     # Determine row separation based on height clustering
#     heights = [y for _, y, _, _ in valid_boxes]
#     median_height = np.median(heights)
#
#     top_row = [box for box in valid_boxes if box[1] < median_height]
#     bottom_row = [box for box in valid_boxes if box[1] >= median_height]
#
#     # Check if it's a **single-row plate** (if all y-values are close together)
#     is_single_row = len(bottom_row) == 0 or abs(top_row[-1][1] - bottom_row[0][1]) < 15
#
#     # Sort characters within each row (left to right)
#     if is_single_row:
#         sorted_boxes = sorted(valid_boxes, key=lambda b: b[0])
#         rows = [sorted_boxes]
#     else:
#         top_row.sort(key=lambda b: b[0])
#         bottom_row.sort(key=lambda b: b[0])
#         rows = [top_row, bottom_row]
#
#     # Recognize characters
#     plate_text = ""
#
#     for row in rows:
#         for x, y, w, h in row:
#             char_crop = eroded[y:y+h, x:x+w]
#             char_crop = cv2.bitwise_not(char_crop)  # Invert for better recognition
#             char_crop = preprocess_character(char_crop)  # Resize & normalize
#
#             prob = model.predict(char_crop)[0]
#             idx = np.argmax(prob)
#             recognized_char = class_names.get(idx, "?")
#
#             plate_text += recognized_char + " " if is_single_row else recognized_char
#
#         plate_text += " " if not is_single_row else ""
#
#     # Debugging visualization (optional)
#     if debug:
#         for (x, y, w, h) in valid_boxes:
#             cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(plate_img, "Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#
#         cv2.imshow("Character Detection", plate_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     return plate_text.strip()
import cv2
import numpy as np
from keras.models import load_model
from utils import preprocess_character

# Load pre-trained character recognition model
model = load_model("D:/ANPR project/code/python/checkpoints/model_epoch_15_.keras")

# Character mapping
class_names = {i: str(i) for i in range(10)}  # Digits 0-9
class_names.update({10: 'ba', 11: 'baa', 12: 'bhe', 13: 'c', 14: 'cha', 15: 'di',
                    16: 'ja', 17: 'ha', 18: 'ja', 19: 'jha', 20: 'ka', 21: 'kha',
                    22: 'ko', 23: 'lu', 24: 'ma', 25: 'me', 26: 'naa', 27: 'nya',
                    28: 'pa', 29: 'pra', 30: 'se', 31: 'su', 32: 'ta', 33: 'ya'})

def recognize_plate_characters(plate_img, debug=False):
    """Detects and recognizes characters from a license plate image."""

    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Apply Otsu's Thresholding for binarization
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Morphological Operations (Erosion) for noise reduction
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)

    # Find contours of characters
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter valid bounding boxes based on size constraints
    valid_boxes = [(x, y, w, h) for (x, y, w, h) in bounding_boxes if 12 < w < 200 and 17 < h < 200]

    if not valid_boxes:
        return "No characters detected"

    # Sort bounding boxes by y-coordinate first (to determine rows)
    valid_boxes.sort(key=lambda b: b[1])

    # Determine row separation based on height clustering
    heights = [y for _, y, _, _ in valid_boxes]
    median_height = np.median(heights)

    top_row = [box for box in valid_boxes if box[1] < median_height]
    bottom_row = [box for box in valid_boxes if box[1] >= median_height]

    # ✅ Fix: Check if `bottom_row` is empty before accessing its elements
    if not bottom_row:
        is_single_row = True
    else:
        is_single_row = abs(top_row[-1][1] - bottom_row[0][1]) < 15 if top_row else True

    # Sort characters within each row (left to right)
    if is_single_row:
        sorted_boxes = sorted(valid_boxes, key=lambda b: b[0])
        rows = [sorted_boxes]
    else:
        top_row.sort(key=lambda b: b[0])
        bottom_row.sort(key=lambda b: b[0])
        rows = [top_row, bottom_row]

    # Recognize characters
    plate_text = ""

    for row in rows:
        for x, y, w, h in row:
            char_crop = eroded[y:y+h, x:x+w]
            char_crop = cv2.bitwise_not(char_crop)  # Invert for better recognition
            char_crop = preprocess_character(char_crop)  # Resize & normalize

            prob = model.predict(char_crop)[0]
            idx = np.argmax(prob)
            recognized_char = class_names.get(idx, "?")

            plate_text += recognized_char + " " if is_single_row else recognized_char

        plate_text += " " if not is_single_row else ""

    return plate_text.strip()

