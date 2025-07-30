import cv2
import imutils
import numpy as np
import argparse

# this function will observe the video in frames and detect any objects (human) and make a box around the person
# it takes a frame and detects a person in it and then returns the frame with the person in a green box
def detect(frame):
    # the detectMultiScale() method returns the coordinates of the box and the confidence value of a person (2-tuple)
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    person = 1
    # creating the green box
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    
    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)

    return frame

# this function takes a path of a video and reads it frame by frame for object detection
def detectByPathVideo(path, writer):

    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while video.isOpened():
        # check if each frame is read successfully, if not then the loop will end
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))   # using imutils to resize each
                                                                            # frame for better computer vision
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

def detectByCamera(writer):   
    video = cv2.VideoCapture(0) # passing 0 in the function means we want to record from webcam
    
    # Check if camera opened successfully
    if not video.isOpened():
        print("Error: Could not open camera.")
        for i in range(1, 5):  # Try camera indices 1-4
            video = cv2.VideoCapture(i)
            if video.isOpened():
                break
        else:
            print("Camera Not Found. Please check your camera connection.")
            return
    
    print('Detecting people...')

    while True:
        check, frame = video.read() # this method reads the video frame by frame
        
        if not check:
            print("Error: Failed to read frame from camera")
            break

        # resize frame before detection for better performance
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        
        frame = detect(frame)   # sending each frame to the detect function, where it will be processed
                                # for object detection 
        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF  # Added & 0xFF for better key detection
        if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

# used when a person needs to be detected from an image
def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    
    if image is None:
        print(f"Image Not Found. Please Enter a Valid Path (Full path of Image Should be Provided).")
        return

    image = imutils.resize(image, width = min(800, image.shape[1])) 

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# this function can take web cameras, videos, and images to detect humans
# if a path is given it will open the video or image in the given path. if not, it will open the web cam
def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']
    
    # More robust camera argument parsing
    camera_arg = str(args["camera"])
    camera = camera_arg == 'True'
    
    print(f"Debug: Camera argument received: '{args['camera']}', parsed as: {camera}")

    writer = None
    if args['output'] is not None and image_path is None:
        writer = cv2.VideoWriter(args['output'],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])
    else:
        print("No input specified. Use -c true for camera, -v for video, or -i for image.")

# will detect arguments in the terminal. -v for video, -i for image, -c for camera, and -o for output (if you want the output to be saved)
def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    # HOGCV is responsible for computer vision and image processing for object detection
    # it is provided by OpenCV, so we can then use this algorithm for object detection
    HOGCV = cv2.HOGDescriptor() # object detection
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = argsParser()
    humanDetector(args)
