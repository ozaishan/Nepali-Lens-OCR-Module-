import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


# Load your trained model (adjust path if needed)
model = load_model('Handwritten_OCR.keras')


def noise_removable(img):
    # resizing the image
    img = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    image_area = img.shape[0] * img.shape[1]

    plt.imshow(img)
    plt.show()

    # converting into grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(img_gray, (3, 3), 0)
    _, thresh_img = cv.threshold(gaussian, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel_er = np.ones((3, 3), np.uint8)
    kernel_di = np.ones((3, 3), np.uint8)
    img_erosion = cv.erode(thresh_img, kernel_er, iterations=1)
    img_dilation = cv.dilate(img_erosion, kernel_di, iterations=1)

    # removing very small noises
    contours, _ = cv.findContours(img_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv.contourArea(contour) < image_area * 0.0005:
            x, y, w, h = cv.boundingRect(contour)
            img_dilation[y:(y + h), x:(x + w)] = 0
    plt.imshow(img_dilation, cmap='gray')
    plt.title("After Threshold + Dilation")
    plt.show()
    # Add contours visualization
    img_contours = cv.cvtColor(img_dilation, cv.COLOR_GRAY2BGR)
    contours, _ = cv.findContours(img_dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    plt.imshow(img_contours)
    plt.title("Contours found")
    plt.show()

    return img_dilation


def ROI(img, axis='horizontal', threshold=5):
    if axis == 'horizontal':
        length = img.shape[0]
        line_dim = img.shape[1]
        line_empty = np.zeros((1, line_dim), np.uint8)
        images_location = []
        line_seg_img = np.array([])

        for r in range(length - 1):
            row_data = img[r:(r + 1), :]
            # Instead of exact zeros, check if sum is less than threshold
            if np.sum(row_data) < threshold or r == length - 2:
                if line_seg_img.size != 0:
                    images_location.append(line_seg_img[:-1])
                    line_seg_img = np.array([])
            else:
                if line_seg_img.size < 1:
                    line_seg_img = row_data
                else:
                    line_seg_img = np.vstack((line_seg_img, img[r + 1:(r + 2), :]))

        return images_location

    elif axis == 'vertical':
        length = img.shape[1]
        col_dim = img.shape[0]
        col_empty = np.zeros((col_dim, 1), np.uint8)
        images_location = []
        word_seg_img = np.array([])

        for c in range(length - 1):
            col_data = img[:, c:(c + 1)]
            if np.sum(col_data) < threshold or c == length - 2:
                if word_seg_img.size != 0:
                    images_location.append(word_seg_img)
                    word_seg_img = np.array([])
            else:
                if word_seg_img.size < 1:
                    word_seg_img = col_data
                else:
                    word_seg_img = np.hstack((word_seg_img, col_data))

        return images_location



def vertical_ROI(img):
    # Vertical ROI segmentation (words)
    row, col = img.shape
    one_col = np.zeros((row, 1), np.uint8)

    images_location = []
    word_seg_img = np.array([])

    for c in range(col - 1):
        if np.array_equal(img[:, c:(c + 1)], one_col) or c == col - 2:
            if word_seg_img.size != 0:
                images_location.append(word_seg_img)
                word_seg_img = np.array([])
        else:
            if word_seg_img.size < 1:
                word_seg_img = img[:, c:(c + 1)]
            else:
                word_seg_img = np.hstack((word_seg_img, img[:, c:(c + 1)]))

    return images_location


def dikka_remove(img):
    # Remove top "dikka" line and segment characters
    template = []
    row, col = img.shape
    image_size = row * col
    img_location = ROI(img)  # horizontal segmentation inside word (usually 1 segment)

    if not img_location:
        return []

    extracted_row = row // 4
    without_dikka = img_location[0][extracted_row:row + 1, :]

    contours, _ = cv.findContours(without_dikka, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours[::-1]:
        if cv.contourArea(contour) > image_size * 0.01:
            x, y, w, h = cv.boundingRect(contour)
            character = ROI(img_location[0][:, x:x + w])
            if character:
                template.append(character[0])

    return template


def preprocessing(img):
    processed_img = noise_removable(img)
    plt.imshow(processed_img, cmap='gray')
    plt.show()

    # Line segmentation
    line_segmentation = ROI(processed_img, axis='horizontal', threshold=10)
    print(f"Total lines found: {len(line_segmentation)}")

    all_words_characters = []

    for line_num, line in enumerate(line_segmentation):
        # Word segmentation using vertical ROI on line image
        print(f"Line {line_num + 1} shape: {line.shape}")
        word_segmentations = ROI(line, axis='vertical', threshold=10)
        print(f"Line {line_num + 1} → Words found: {len(word_segmentations)}")

        for word_num, word in enumerate(word_segmentations):
            
            plt.imshow(word, cmap='gray')
            plt.title(f"Line {line_num + 1}, Word {word_num + 1}")
            plt.axis('off')
            plt.show()

            characters = dikka_remove(word)
            if characters:
                all_words_characters.append(characters)

                plt.imshow(characters[0], cmap='gray')
                plt.title(f"First character of Word {word_num + 1}")
                plt.axis('off')
                plt.show()

    return all_words_characters


def prediction(final_segmented_img):
    words = len(final_segmented_img)
    accuracy = []

    characters_list = 'ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,क,न,प,फ,ब,भ,म,य,र,ल,व,ख,श,ष,स,ह,क्ष,त्र,ज्ञ,ग,घ,ङ,च,छ,ज,झ,०,१,२,३,४,५,६,७,८,९'
    characters = characters_list.split(',')

    for i in range(words):
        chars = final_segmented_img[i]
        for img in chars:
            img = np.pad(img, ((2, 2), (2, 2)), mode='constant')  # Padding (top/bottom, left/right)
            img = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA)
            x = np.asarray(img, dtype=np.float32).reshape(1, 32, 32, 1) / 255.0

            output = model.predict(x)
            output = output.reshape(46)
            predicted = np.argmax(output)
            devanagari_label = characters[predicted]

            accuracy.append(output[predicted] * 100)
            print(devanagari_label, end='')  # Print character

        print(' ', end='')  # Space after each word

    return np.mean(accuracy)


# --- USAGE ---

img = cv.imread('OCR_test.jpg')
final_segmented_img = preprocessing(img)
avg_accuracy = prediction(final_segmented_img)
print(f"\nAverage Prediction Accuracy: {avg_accuracy:.2f}%")
