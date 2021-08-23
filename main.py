import cv2

thres = 0.5 # Threshold to deteect
#img = cv2.imread('resources/img.PNG')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile = 'class_file_goes_here'
with open(classFile, 'rt') as f:
	classNames = f.read().strip('\n').split('\n')

#Trained weight and config
configPath = 'weights_config_goes_here'
weightsPath = 'weights_goes_here'

#Model

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
	success,img = cap.read()
	#img = cv2.imread('resources/cat.png')
	classIds, confs, bbox = net.detect(img, confThreshold=thres)
	print(classIds, bbox)

	if len(classIds) != 0:
		for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
			cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
			#cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5,
			            #(0, 255, 0), 1)
			cv2.putText(img, classNames[classId - 1].upper(), (box[0], box[1] - 10),
		cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)


		cv2.putText(img, str(confidence), (box[0] + 10, box[1] + 28), cv2.FONT_HERSHEY_COMPLEX, 0.5,
			            (0, 255, 0), 1)

	cv2.imshow('Output', img)
	cv2.waitKey(1)
