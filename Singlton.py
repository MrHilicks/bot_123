from imageai.Detection import ObjectDetection
import os


class CameraCheck:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not CameraCheck._instance:
            CameraCheck._instance = super(CameraCheck, cls).__new__(cls, *args, **kwargs)
        return CameraCheck._instance

    def __init__(self):
        self.detector = ObjectDetection()
        # self.detector.setModelTypeAsTinyYOLOv3()
        # self.detector.setModelPath("yolo-tiny.h5")
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath("yolo.h5")
        self.detector.loadModel()
        self.custom = self.detector.CustomObjects(person=True)

    def Check(self, image_name):
        execution_path = os.getcwd()
        detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom,
                                                                input_image=os.path.join(execution_path,
                                                                                         image_name),
                                                                output_image_path=os.path.join(execution_path,
                                                                                               "detected_people.jpg"),
                                                                minimum_percentage_probability=20)
        return len(detections)


if __name__ == '__main__':
    bot = CameraCheck()
    print(bot.Check('image.jpg'))
    print(bot.Check('image1.jpg'))
    print(bot.Check('image2.jpg'))
    print(bot.Check('image3.jpg'))
