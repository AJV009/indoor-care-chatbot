import cv2
import numpy as np
import time
import collections
from IPython import display

from utils.videoPlayer import VideoPlayer
from utils.postProcess import detect_without_preprocess, draw_results

# Generated some rough real world sizes for the objects for simple rough approx depth estimation (using GPT-4)
coco_real_world_sizes = {
    0: {"person": {"height": [1.5, 2.0], "width": [0.5, 0.7]}},
    1: {"bicycle": {"height": [1.0, 1.5], "width": [1.5, 2.0]}},
    2: {"car": {"height": [1.4, 1.6], "width": [1.8, 2.2]}},
    3: {"motorcycle": {"height": [1.0, 1.5], "width": [1.5, 2.0]}},
    4: {"airplane": {"height": [10.0, 60.0], "width": [30.0, 80.0]}},
    5: {"bus": {"height": [3.0, 4.5], "width": [2.5, 3.0]}},
    6: {"train": {"height": [4.0, 5.0], "width": [3.0, 3.5]}},
    7: {"truck": {"height": [3.5, 4.5], "width": [2.5, 3.0]}},
    8: {"boat": {"height": [2.5, 20.0], "width": [2.5, 20.0]}},  # Varies greatly
    9: {"traffic light": {"height": [2.0, 4.0], "width": [0.3, 0.5]}},
    10: {"fire hydrant": {"height": [0.7, 1.0], "width": [0.3, 0.5]}},
    11: {"stop sign": {"height": [1.5, 1.8], "width": [0.6, 0.6]}},
    12: {"parking meter": {"height": [1.0, 1.5], "width": [0.3, 0.3]}},
    13: {"bench": {"height": [0.8, 1.2], "width": [1.2, 2.0]}},
    14: {"bird": {"height": [0.05, 0.5], "width": [0.05, 0.5]}},  # Varies greatly
    15: {"cat": {"height": [0.2, 0.3], "width": [0.2, 0.5]}},
    16: {"dog": {"height": [0.3, 1.0], "width": [0.3, 1.0]}},  # Varies greatly
    17: {"horse": {"height": [2.0, 2.5], "width": [1.0, 2.5]}},
    18: {"sheep": {"height": [0.8, 1.0], "width": [0.5, 1.0]}},
    19: {"cow": {"height": [1.4, 1.9], "width": [1.8, 2.2]}},
    20: {"elephant": {"height": [2.5, 4.0], "width": [3.0, 5.0]}},
    21: {"bear": {"height": [1.0, 3.0], "width": [0.75, 2.0]}},  # Varies greatly
    22: {"zebra": {"height": [1.2, 1.5], "width": [1.0, 2.5]}},
    23: {"giraffe": {"height": [4.5, 6.0], "width": [1.5, 2.0]}},
    24: {"backpack": {"height": [0.4, 0.6], "width": [0.3, 0.4]}},
    25: {"umbrella": {"height": [1.0, 1.5], "width": [1.0, 1.5]}},  # when opened
    26: {"handbag": {"height": [0.2, 0.4], "width": [0.3, 0.5]}},
    27: {"tie": {"height": [1.3, 1.5], "width": [0.1, 0.15]}},
    28: {"suitcase": {"height": [0.55, 0.85], "width": [0.35, 0.55]}},
    29: {"frisbee": {"height": [0.025, 0.03], "width": [0.21, 0.27]}},
    30: {"skis": {"height": [1.5, 2.2], "width": [0.1, 0.2]}},
    31: {"snowboard": {"height": [1.3, 1.6], "width": [0.25, 0.3]}},
    32: {"sports ball": {"height": [0.2, 0.3], "width": [0.2, 0.3]}},  # Varies greatly
    33: {"kite": {"height": [0.5, 2.0], "width": [0.5, 2.0]}},  # Varies greatly
    34: {"baseball bat": {"height": [0.7, 1.0], "width": [0.03, 0.06]}},
    35: {"baseball glove": {"height": [0.2, 0.3], "width": [0.2, 0.3]}},
    36: {"skateboard": {"height": [0.1, 0.15], "width": [0.2, 0.8]}},
    37: {"surfboard": {"height": [1.7, 2.8], "width": [0.45, 0.6]}},
    38: {"tennis racket": {"height": [0.68, 0.73], "width": [0.23, 0.29]}},
    39: {"bottle": {"height": [0.2, 0.33], "width": [0.06, 0.1]}},
    40: {"wine glass": {"height": [0.15, 0.25], "width": [0.07, 0.10]}},
    41: {"cup": {"height": [0.10, 0.15], "width": [0.08, 0.12]}},
    42: {"fork": {"height": [0.15, 0.20], "width": [0.02, 0.03]}},
    43: {"knife": {"height": [0.20, 0.25], "width": [0.02, 0.03]}},
    44: {"spoon": {"height": [0.15, 0.20], "width": [0.03, 0.05]}},
    45: {"bowl": {"height": [0.07, 0.15], "width": [0.15, 0.30]}},
    46: {"banana": {"height": [0.15, 0.25], "width": [0.03, 0.05]}},
    47: {"apple": {"height": [0.07, 0.10], "width": [0.07, 0.10]}},
    48: {"sandwich": {"height": [0.05, 0.10], "width": [0.10, 0.15]}},
    49: {"orange": {"height": [0.07, 0.10], "width": [0.07, 0.10]}},
    50: {"broccoli": {"height": [0.1, 0.3], "width": [0.1, 0.3]}},
    51: {"carrot": {"height": [0.2, 0.3], "width": [0.03, 0.05]}},
    52: {"hot dog": {"height": [0.03, 0.05], "width": [0.15, 0.2]}},
    53: {"pizza": {"height": [0.02, 0.03], "width": [0.3, 0.5]}},
    54: {"donut": {"height": [0.05, 0.1], "width": [0.1, 0.2]}},
    55: {"cake": {"height": [0.1, 0.2], "width": [0.2, 0.4]}},
    56: {"chair": {"height": [1, 1.2], "width": [0.5, 0.7]}},
    57: {"couch": {"height": [0.8, 1.0], "width": [1.5, 2.5]}},
    58: {"potted plant": {"height": [0.2, 1.5], "width": [0.2, 1.0]}},  # Varies greatly
    59: {"bed": {"height": [0.5, 1.0], "width": [1.0, 2.0]}},
    60: {"dining table": {"height": [0.7, 1.0], "width": [1.0, 2.2]}},
    61: {"toilet": {"height": [0.4, 0.5], "width": [0.4, 0.5]}},
    62: {"tv": {"height": [0.3, 2.0], "width": [0.5, 3.0]}},  # Can vary greatly based on the size of the TV
    63: {"laptop": {"height": [0.02, 0.05], "width": [0.3, 0.45]}},
    64: {"mouse": {"height": [0.03, 0.05], "width": [0.06, 0.1]}},
    65: {"remote": {"height": [0.02, 0.03], "width": [0.05, 0.2]}},
    66: {"keyboard": {"height": [0.02, 0.05], "width": [0.2, 0.5]}},
    67: {"cell phone": {"height": [0.014, 0.02], "width": [0.07, 0.09]}},
    68: {"microwave": {"height": [0.3, 0.6], "width": [0.5, 0.8]}},
    69: {"oven": {"height": [0.9, 1.5], "width": [0.6, 0.9]}},
    70: {"toaster": {"height": [0.2, 0.3], "width": [0.2, 0.3]}},
    71: {"sink": {"height": [0.8, 1.0], "width": [0.5, 1.0]}},
    72: {"refrigerator": {"height": [1.5, 2.0], "width": [0.6, 1.0]}},
    73: {"book": {"height": [0.02, 0.04], "width": [0.15, 0.3]}},
    74: {"clock": {"height": [0.2, 0.4], "width": [0.2, 0.4]}},  # Wall clocks; wrist watches would be smaller
    75: {"vase": {"height": [0.3, 1.0], "width": [0.1, 0.3]}},
    76: {"scissors": {"height": [0.05, 0.15], "width": [0.1, 0.2]}},
    77: {"teddy bear": {"height": [0.3, 1.0], "width": [0.2, 0.5]}},  # Varies greatly
    78: {"hair drier": {"height": [0.1, 0.2], "width": [0.1, 0.3]}},
    79: {"toothbrush": {"height": [0.01, 0.02], "width": [0.1, 0.15]}},
}


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
            
            image_with_boxes, distance_array = draw_results(detections, input_image, label_map, coco_real_world_sizes)
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

    # Extract unique distance arrays
    distance_arrays = [x for x in distance_arrays if x != []]
    flat_list = [item for sublist in distance_arrays for item in sublist]
    dct = {(item[0], item[1]): item for item in reversed(flat_list)}
    unique_distance_arrays = list(dct.values())

    # Store all unique objects
    all_detected_objects = set()
    for item in unique_distance_arrays:
        all_detected_objects.add(item[0])
        all_detected_objects.add(item[1])

    # Return the list of distance arrays and the set of all detected objects
    return unique_distance_arrays, all_detected_objects

