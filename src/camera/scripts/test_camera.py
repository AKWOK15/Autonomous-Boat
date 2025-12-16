import cv2

# Open camera using the same device path
# cap = cv2.VideoCapture('/dev/video0')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)


# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the frame
    cv2.imshow('Camera Feed', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()