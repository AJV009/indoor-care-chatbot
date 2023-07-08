# YOLOv8 OpenVINO Optimized Demo
This is a demo of YOLOv8 object detection model optimized with OpenVINO Toolkit.

## How to run
1. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
2. Create the optimized model by running the notebook cells at `convert.ipynb` to convert the model to OpenVINO format. **(IMPORTANT)**
2. Run the demo:
    ```bash
    python demo.py
    ```
    **OR** You can run the notebook version of the demo from `demo.ipynb` file.
    
_Note: Demo won't work without creating the optimized model._

## Demo (CPU)
![demo2](demoImages/demo2.png)
