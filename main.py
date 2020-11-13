from imageai.Detection import ObjectDetection
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("yolo-tiny.h5")
detector.loadModel(detection_speed='fastest')
custom = detector.CustomObjects(person=True)

detections = detector.detectCustomObjectsFromImage(
    custom_objects=custom,
    input_image=os.path.join(execution_path, "image.jpg"),
    output_image_path=os.path.join(execution_path, "detected_people.jpg"),
    minimum_percentage_probability=20)

persons = len(detections)
print('Количество людей в очереди: ' + str(persons))
