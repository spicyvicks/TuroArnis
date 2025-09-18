# import necessary libraries
import cv2
from ultralytics import YOLO

# --- setup ---
# load the official yolov8 nano model.
# the '.pt' file will be downloaded automatically the first time you run this.
# 'yolov8n.pt' is small and fast, perfect for real-time detection.
model = YOLO('yolov8n.pt')

# --- video source ---
# use 0 for your primary webcam.
# or, you can replace 0 with a video file path, e.g., 'my_video.mp4'
cap = cv2.VideoCapture(0)

# check if the webcam opened successfully
if not cap.isOpened():
    print("error: could not open video source.")
    exit()

# --- main loop to process video frames ---
while cap.isOpened():
    # read a single frame from the video source
    success, frame = cap.read()

    # if a frame was successfully read
    if success:
        # --- yolo detection ---
        # run the yolo model on the frame.
        # 'stream=true' is efficient for processing video feeds.
        results = model(frame, stream=True)

        # loop through all the results detected in the frame
        for r in results:
            # get the bounding boxes for each detected object
            boxes = r.boxes

            # loop through each individual box
            for box in boxes:
                # --- filter for 'person' class ---
                # the 'person' class in the coco dataset (which yolo was trained on) is class id 0.
                # we only want to process detections that are people.
                if box.cls[0] == 0: # class 0 is 'person'
                    
                    # get the coordinates of the bounding box
                    # box.xyxy[0] gives coordinates in the format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to integers

                    # get the confidence score (how sure the model is)
                    confidence = float(box.conf[0])

                    # --- draw on the frame ---
                    # draw the rectangle around the detected person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3) # draw a purple box

                    # create the label text with class name and confidence
                    label = f'person {confidence:.2f}'

                    # draw the label text above the box
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        # display the final frame with all the drawings
        cv2.imshow("yolo real-time person detection", frame)

        # break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    else:
        # break the loop if we reach the end of the video file
        break

# --- cleanup ---
# release the video capture object and close all opencv windows
cap.release()
cv2.destroyAllWindows()