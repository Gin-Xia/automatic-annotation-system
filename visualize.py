import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_results(image_np, best_box, masks, category_name, confidence):
    """
    Visualize bounding boxes and masks on an image.

    Args:
        image_np (numpy array): The image in NumPy format.
        best_box (numpy array): The predicted bounding box in pixel coordinates.
        masks (numpy array): The predicted mask from MobileSAM.
        category_name (str): The detected object category.
        confidence (float): The confidence score.
    """
    # Convert float32 image to uint8
    image_np = (image_np * 255).astype(np.uint8)

    # Convert grayscale mask to 3-channel
    mask_overlay = (masks[0] > 0.5).astype(np.uint8) * 255

    # Resize mask to match the image
    mask_overlay = cv2.resize(mask_overlay, (image_np.shape[1], image_np.shape[0]))

    # Apply color map to the mask
    mask_colored = cv2.applyColorMap(mask_overlay, cv2.COLORMAP_JET).astype(np.uint8)

    # Blend the mask with the image
    blended = cv2.addWeighted(image_np, 0.7, mask_colored, 0.3, 0)  # Ensure both are uint8

    # Draw bounding box
    x, y, w, h = map(int, best_box[0])
    cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    # Add label
    label = f"{category_name} ({confidence:.2f})"
    cv2.putText(blended, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display image with bounding box and mask
    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Object Detection & Segmentation Result")
    plt.show()
