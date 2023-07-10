import cv2
import numpy as np
import time
import collections
from IPython import display

from utils.videoPlayer import VideoPlayer
from utils.preprocess import detect_without_preprocess, draw_results

def run_object_detection(source=0, flip=False, skip_first_frames=0, model="None", label_map={}, core='', device="None", interval=10):
    player = None
    # Convert interval from ms to seconds
    interval /= 1000
    
    # Array to hold the distance arrays
    distance_arrays = []
    
    compiled_model = core.compile_model(model, device)
    try:
        player = VideoPlayer(
            source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames
        )
        player.start()

        processing_times = collections.deque()
        
        # Record start time
        start_time_interval = time.time()
        
        while True:
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            input_image = np.array(frame)
           
            start_time = time.time()
            detections = detect_without_preprocess(input_image, compiled_model)[0]
            stop_time = time.time()
            
            image_with_boxes, distance_array = draw_results(detections, input_image, label_map)
            frame = image_with_boxes
           
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            # Add distance_array to the list
            distance_arrays.append(distance_array)

            _, f_width = frame.shape[:2]

            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            # Encode numpy array to jpg.
            _, encoded_img = cv2.imencode(
                ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
            )
            # Create an IPython image.
            i = display.Image(data=encoded_img)
            # Display the image in this notebook.
            display.clear_output(wait=True)
            display.display(i)
            
            # If the time elapsed is more than the interval, stop the loop
            if time.time() - start_time_interval > interval:
                break
                
    except RuntimeError as e:
        print(e)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if player is not None:
            player.stop()
    # remove empty arrays
    distance_arrays = [x for x in distance_arrays if x != []]
    # Store all unique objects
    flattened_distance_arrays = [item for sublist in distance_arrays for item in sublist]
    objects = [item[0:2] for item in flattened_distance_arrays]
    flattened_objects = [item for sublist in objects for item in sublist]
    all_detected_objects = list(set(flattened_objects))
    # Return the list of distance arrays
    return distance_arrays, all_detected_objects

