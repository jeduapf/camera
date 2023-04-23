import cv2

def config_IP_cam():
    cam_rtsp = 'rtsp://admin:Pai39mae39@192.168.0.74:554/cam/realmonitor?channel=1&subtype=0'
    # cam_https = 'https://admin:Pai39mae39@192.168.0.74:443/cam/realmonitor?channel=1&subtype=0' #[tcp @ 000001fef27a3b80] Connection to tcp://192.168.0.74:443 failed: Error number -138 occurred
    # cam_http = 'https://admin:Pai39mae39@192.168.0.74:80/cam/realmonitor?channel=1&subtype=0' #[tls @ 000001e6913f3480] Failed to read handshake response

    cap = cv2.VideoCapture(cam_rtsp)
    
    if cap is None or not cap.isOpened():
        print(f'Warning: unable to open video source: {cam_rtsp}')
    
    video_info_dict = {
        "FPS" : cap.get(cv2.CAP_PROP_FPS), 
        "HEIGTH" : cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "WIDTH" : cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "MODE" : cap.get(cv2.CAP_PROP_MODE),
        "FORMAT" : cap.get(cv2.CAP_PROP_FORMAT), 
        "BUFFERSIZE" : cap.get(cv2.CAP_PROP_BUFFERSIZE), 
        "CHANNEL" : cap.get(cv2.CAP_PROP_CHANNEL), 
        "BITRATE" : cap.get(cv2.CAP_PROP_BITRATE),       
    }
    
    return cap, video_info_dict

def webcam_face_detection(cap):   
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    while True:
        # Read the frame
        ret, img = cap.read()
        # gpu_frame = cv2.cuda_GpuMat()
        
        if ret:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            detections = len(faces)
            if detections > 0 :
                print(f"Pessoas detectadas: {detections}")
                
                # Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            # Display
            cv2.imshow('img', img)
            
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
    # Release the VideoCapture object
    cap.release()
    
def main():
    # webcam_face_detection()
    # hog_detector()
    # print(cv2.cuda.getCudaEnabledDeviceCount())
    
    cap, video_info_dict = config_IP_cam()
    print(video_info_dict)
    webcam_face_detection(cap)
    
    
if __name__ == "__main__":
    main()
