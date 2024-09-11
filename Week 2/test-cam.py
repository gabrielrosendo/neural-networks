import cv2
import numpy as np

def detect_black(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to get the black areas
    _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded

def main():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Detect black areas in the frame
        black_areas = detect_black(frame)
        
        # Display the original frame and the thresholded frame
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Black Areas', black_areas)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()