# Facial Emotion Recognition using Deepface and Haarcascade algorithm

## Libraries

- OpenCV
- pickle
- DeepFace

## Face Detection using Haarcascade
The Haarcascade classifier is a pre-trained XML file used to detect faces in images or video frames.This following line initializes the Haarcascade face detection model.

```python
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
```
## Video Capture
The code below tries to open the webcam using OpenCV on different devices (indices 0 or 1).
```python
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam.")
```
## Real-Time Emotion Detection
The loop captures frames from the webcam and processes them in real time.The DeepFace.analyze() method analyzes the frame for emotions and returns a dictionary with dominant_emotion and probabilities for all detected emotions.
```python
ret, frame = cap.read()
result = Deepface.analyze(frame, actions=['emotion'])
```
## Face Detection and Rectangle Drawing
This part:

- Converts the frame to grayscale.
- Detects faces using faceCascade.detectMultiScale().
- Draws a rectangle around each detected face.

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)
for x, y, w, h in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```
## Emotion Display
- The dominant emotion from DeepFace is displayed on the video frame.
```python
 font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(frame, result['dominant_emotion'], (10, 10), font, 1, (0, 255, 255), 2)
```
## Emotion Frequency Count
- This dictionary d keeps track of how many times each emotion is detected.
```python
if result["dominant_emotion"] not in d:
    d[result["dominant_emotion"]] = 1
else:
    d[result["dominant_emotion"]] += 1
```

## Saving and Loading the Model
- Saves the DeepFace result to a file named facemodel.pkl.
- Later reloads the data for future use.
```python
pickle.dump(result, open("facemodel.pkl", "wb"))
model = pickle.load(open("facemodel.pkl", "rb"))
```
## Most Frequent Emotion
After analyzing 30 frames, the emotion with the highest count is printed.
```python
key_with_max_value = max(d, key=d.get)
print(key_with_max_value)
```
## Cleanup
Finally, the webcam is released, and all OpenCV windows are closed:
```python
cap.release()
cv2.destroyAllWindows()
```
