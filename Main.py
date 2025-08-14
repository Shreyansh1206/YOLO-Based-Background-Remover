from ultralytics import YOLO
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

model = YOLO('yolov8n-seg.pt')

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read() 
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for result in results:
        masks = result.masks
        if masks is None:
            continue
        for m in masks.data:
            mask = m.cpu().numpy().astype(np.uint8) * 255
            final_mask = cv.bitwise_or(final_mask, mask)

    masked_frame = cv.bitwise_and(frame, frame, mask=final_mask)

    bgra = cv.cvtColor(masked_frame, cv.COLOR_BGR2BGRA)
    bgra[:, :, 3] = final_mask  # alpha channel from mask

    cv.imshow('Masked Frame', bgra)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
