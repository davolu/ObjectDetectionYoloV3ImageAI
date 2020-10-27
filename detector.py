#inspired by https://stackabuse.com/object-detection-with-imageai-in-python/
#sample image gotten from https://pixabay.com/photos/traffic-jam-traffic-india-street-388924/

from imageai.Detection import ObjectDetection

detectorObj = ObjectDetection()
model_h5_path = "./models/yolo-tiny.h5"
input_file_path = "./input/traffic-jam-388924_640.jpg"
output_file_path = "./output/output.jpg"


detectorObj.setModelTypeAsTinyYOLOv3()
detectorObj.setModelPath(model_h5_path)
detectorObj.loadModel()

detectionResult = detectorObj.detectObjectsFromImage(input_image=input_file_path, output_image_path=output_file_path)

for DetectedItem in detectionResult:
    print(DetectedItem["name"] , " : ", DetectedItem["percentage_probability"])

 