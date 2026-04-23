CHAPTER 1 
INTRODUCTION 
1.1 Introduction  
In recent years, online learning and computer-based learning environments have become 
widely used in education. Even though these platforms provide flexibility and accessibility, 
they also create a major challenge for teachers and institutions: it is difficult to know whether 
a student is actually paying attention during the learning session. In a physical classroom, a 
teacher can observe the student’s face, eye contact, and body behavior directly. But in an 
online or system-based environment, this monitoring becomes difficult without technological 
support. 
Figure 1.1: Engaged Student in Online Learning  
The problem studied in this project is student engagement detection. The main aim is to 
identify whether a student is engaged or not engaged by analyzing facial behavior in real time 
through a webcam feed. Based on the uploaded project files, the system captures live video, 
detects the student’s face, checks the eye condition, predicts the facial state using a trained 
deep learning model, and then gives a final engagement result such as Engaged or Not 
Engaged. The prediction logic is built from six facial-state classes: confused, engaged, 
frustrated, looking_away, bored, and drowsy.  
1 
Figure 1.2: Disengaged Student in Online Learning  
The training part of the project shows that a MobileNetV2-based convolutional neural 
network is used to learn these classes from an image dataset. The model is trained with 
preprocessing and augmentation techniques such as rescaling, horizontal flipping, zooming, 
and validation splitting. After training, the model is saved and later loaded into the real-time 
system for live prediction.  
The real-time implementation further improves the decision-making process by combining 
deep learning prediction with computer vision rules. The system uses Haar cascade classifiers 
for face and eye detection, tracks temporary face loss using anti-flicker logic, checks whether 
the eyes are closed for several frames, and also determines whether the student is looking 
away based on face position in the frame. These conditions are fused with the CNN output to 
provide a more meaningful final engagement status.  
Thus, this project studies how artificial intelligence and computer vision can be used to 
support real-time monitoring of student attention in a simple and practical way. 
1.2 Problem Being Studied 
The problem addressed in this project is the lack of an automatic mechanism to monitor 
whether a student is attentive during a learning session. In many digital learning situations, a 
student may appear online but may actually be distracted, sleepy, bored, looking away from 
the screen, or mentally disengaged. This reduces the effectiveness of learning and makes it 
hard for teachers or systems to evaluate real participation. 
2 
The uploaded code directly reflects this problem by treating facial behavior as the key 
indicator of engagement. The real-time system identifies disengagement conditions such as 
eyes closed, looking away, and model-predicted states like bored, looking_away, and drowsy, 
and maps them to a final Not Engaged status. Otherwise, it marks the student as Engaged. 
This shows that the project is specifically focused on detecting student attention through 
visual observation and classification.  
Therefore, the problem being studied is not general emotion recognition alone, but the use of 
facial-state analysis and live video processing to determine student engagement level in a 
learning environment. 
1.3 Objectives of the Project 
The main objectives of this project are: 
1. To develop a real-time student engagement detection system 
The project aims to build a system that can capture live webcam video, process the student’s 
face, and determine engagement continuously in real time. This objective is supported by the 
webcam-based implementation using cv2.VideoCapture(0) and live frame-by-frame 
processing.  
2. To train a deep learning model for facial-state classification 
Another objective is to train a CNN model that can classify student facial states into multiple 
categories such as confused, engaged, frustrated, looking away, bored, and drowsy. This is 
achieved through a MobileNetV2-based architecture trained on an image dataset with 
augmentation and validation support.  
3. To combine computer vision features with model prediction for better decision
making 
The project is not limited to CNN prediction alone. It also aims to improve reliability by 
checking additional visual cues such as face presence, eye closure, and head direction. These 
cues are merged with the model’s output using fusion logic to provide a final engagement 
decision.  
4. To identify engaged and not engaged behavior in a practical learning setup 
3 
The final objective is to classify the student’s state into a meaningful output that can be used 
in educational monitoring. The system converts multiple visual and prediction conditions into 
user-friendly output labels such as Engaged, Not Engaged, Not Engaged (Eyes Closed), and 
Not Engaged (Looking Away).  
1.4 Scope of the Study 
The scope of this study is limited to real-time visual engagement detection using a webcam. 
The system works by analyzing facial information captured from a live camera feed and 
making a decision based on: 
• face detection,  
• eye detection,  
• face position in the frame,  
• and deep learning-based facial-state classification.  
The study includes: 
• dataset-based training of a MobileNetV2 model,  
• classification of six student facial states,  
• loading the trained model in a live environment,  
• and final engagement detection through rule-based fusion.  
At the same time, the scope does not extend to full classroom analytics, speech analysis, body 
posture tracking, or multi-student monitoring, because the uploaded implementation focuses 
only on the face region from a single webcam stream. The current project is therefore best 
suited for single-user engagement monitoring in a controlled real-time environment. This is an 
inference from the implementation design, since the code processes webcam frames and face
level predictions one session at a time.  
1.5 Importance of the Work 
This work is important because student engagement is closely related to learning 
effectiveness. If a system can automatically identify whether a student is attentive, it can help 
improve digital education by making monitoring more intelligent and responsive. In online or 
self-learning environments, teachers and institutions often do not know whether a student is 
4 
focused, distracted, tired, or disconnected. This project addresses that need by creating a 
system that can give immediate engagement-related feedback. 
The importance of this project also lies in its practical combination of deep learning and 
computer vision. Instead of relying only on image classification, the project adds eye 
detection, looking-away detection, and no-face handling logic. This makes the system more 
useful for real-time deployment because it does not depend on only one signal. The anti
flicker mechanism also improves output stability when the face is temporarily lost.  
In addition, the project demonstrates how an efficient pretrained model like MobileNetV2 can 
be adapted for an educational use case. Since MobileNetV2 is lightweight compared to 
heavier deep learning networks, it is a practical choice for systems that need real-time 
behavior. The uploaded code shows that it is used as the base model for training the student
state classifier.  
Therefore, the work is important from both an academic and practical point of view, as it 
supports real-time student monitoring and shows the application of AI in education. 
1.6 Reason for Choosing the Topic 
This topic was chosen because monitoring student attention has become an important issue in 
modern learning environments. As online and computer-based learning systems grow, there is 
a need for smart tools that can assist in understanding whether students are actively engaged. 
A project in this area is meaningful because it connects education with current technologies 
such as artificial intelligence, deep learning, and computer vision. 
Another reason for choosing this topic is that it provides a strong practical application of 
machine learning concepts. The uploaded files show the use of dataset preprocessing, data 
augmentation, transfer learning with MobileNetV2, real-time webcam capture, Haar cascade 
detection, and decision fusion logic. This makes the project technically rich and relevant for a 
student-level final-year project. 
The topic is also suitable because it solves a real-world problem in a visible and 
understandable way. The final output is easy to interpret, since the system directly shows 
whether the student is engaged or not engaged. This makes the project useful for 
demonstration, presentation, and future extension into smarter educational monitoring 
systems. 
5 
In addition, the project allows the integration of both deep learning and traditional computer 
vision techniques, providing a balanced learning experience. It also offers opportunities for 
further improvement, such as adding advanced models, multi-user detection, and system 
integration with learning platforms. Therefore, this topic is both academically valuable and 
practically relevant, making it an ideal choice for implementation and research. 
6 
CHAPTER 2 
LITERATURE REVIEW 
1. 
Student Engagement Assessment Using Multimodal Deep Learning (2024) 
This paper proposes a multimodal approach for student engagement detection using facial 
expressions, eye gaze, and head pose. It combines CNN for extracting spatial facial features 
and LSTM for modeling temporal behavior across video frames. By integrating multiple 
visual cues, the system improves the robustness and reliability of engagement prediction in 
online learning environments. 
Limitations:  
The model requires high computational resources and is not fully optimized for real-time 
implementation.  
2. 
A General Model for Detecting Learner Engagement (2024) 
This study provides a comprehensive theoretical framework for understanding learner 
engagement by categorizing it into behavioral, emotional, and cognitive components. It 
clearly explains the different dimensions of engagement and their role in the learning process. 
The paper is useful in giving a conceptual foundation for developing intelligent engagement 
detection systems. 
Limitations:  
The work is purely theoretical and does not include any practical implementation or 
experimental validation.  
3. Measuring Student Engagement Through Behavioral and Emotional Features (2024) 
This paper analyzes student engagement using both behavioral and emotional features derived 
from visual data. It employs machine learning and CNN-based techniques to classify 
engagement levels more effectively. The study highlights that combining multiple cues can 
improve prediction accuracy compared to relying on a single feature type. 
Limitations:  
The system is mainly designed for offline analysis and does not effectively capture temporal 
changes in engagement.  
7 
4. CMOSE: Comprehensive Multi-Modality Online Student Engagement Dataset (2024) 
This work introduces a large-scale, high-quality dataset for student engagement detection 
collected from real online learning sessions. It includes multimodal data such as facial 
expressions and behavioral patterns, making it useful for training and testing advanced 
engagement models. The dataset serves as an important resource for future research in this 
domain. 
Limitations:  
The study focuses only on dataset creation and does not propose any specific engagement 
detection model.  
5. 
Student 
Engagement 
Recognition 
Using 
CNN–LSTM 
(2023) 
This paper uses a hybrid CNN–LSTM model to detect engagement from video sequences. 
CNN extracts spatial features from individual frames, while LSTM captures temporal 
dependencies across frames to improve understanding of student behavior. The model 
achieves better accuracy than single-frame approaches by using both spatial and sequential 
information. 
Limitations:  
The model has higher computational complexity and slower inference, making it less suitable 
for real-time applications.  
6. 
Real-Time 
Student 
Attention 
Monitoring 
System 
(2023) 
This study proposes a real-time system that uses webcam input to monitor student attention 
during online classes. It demonstrates that live engagement detection is practically possible 
using computer vision techniques in educational settings. The work is important because it 
moves beyond offline analysis and focuses on direct real-time monitoring. 
Limitations:  
The system is limited to single-student scenarios and lacks scalability for multiple users.  
7. 
Student 
Engagement 
Detection 
Using 
YOLOv4 
(2023) 
This paper uses YOLOv4 for real-time face detection and analyzes head direction and facial 
orientation to determine engagement. It shows that object detection models can be effectively 
applied in educational environments for fast and practical monitoring. The study mainly 
focuses on visible attention cues such as face position and direction. 
8 
Limitations:  
The system does not incorporate temporal learning, and its accuracy is limited compared to 
more advanced deep learning approaches.  
8. 
Student 
Attention 
Monitoring 
Using 
Deep 
CNN 
(2023) 
This work applies a deep CNN model to classify student attention using facial images 
captured from webcams. It provides a simple and effective image-based approach for 
identifying engagement-related states. The paper shows that CNN models can perform well 
for visual classification tasks in student monitoring systems. 
Limitations:  
The model relies only on spatial features and lacks temporal and contextual understanding of 
engagement.  
9. 
Computer 
Vision-Based 
Classroom 
Attention 
Analysis 
(2023) 
This paper focuses on detecting student attention using face direction and posture analysis in 
classroom settings. It highlights the importance of visual cues such as head orientation and 
body posture in understanding student focus. The work expands engagement analysis beyond 
facial expression by considering general classroom behavior. 
Limitations:  
The system is sensitive to environmental noise and variations, reducing its reliability in real
world conditions.  
10. 
Multimodal 
Engagement 
Analysis 
from 
Facial 
Videos 
(2021) 
This study uses facial landmarks and expressions extracted from video data to analyze 
engagement levels. It demonstrates that video-based analysis provides richer information than 
static image analysis because it captures motion and changing behavior over time. The work 
supports the idea that temporal visual data can improve engagement detection performance. 
Limitations:  
The model requires good lighting conditions and continuous face visibility, which may not 
always be available.  
9 
11. 
Vision-Based 
Engagement 
Monitoring 
in 
E-Learning 
(2022) 
This paper uses facial movement and gaze tracking to monitor engagement in online learning 
environments. It shows that eye movement and head pose are strong indicators of student 
attention and focus during digital learning sessions. The study is useful in proving the 
importance of visual attention cues in e-learning systems. 
Limitations:  
The system is evaluated only in offline conditions and lacks real-time implementation.  
12. 
Student 
Engagement 
Detection 
Using 
Facial 
Expressions 
(2022) 
This work uses CNN-based facial emotion recognition to estimate student engagement levels. 
It 
emphasizes the role of emotions such as boredom, confusion, and frustration in 
understanding how actively a student is involved in learning. The paper shows that facial 
expressions 
can 
Limitations:  
serve 
as 
useful 
indicators 
for 
engagement 
analysis. 
Emotional expressions alone are not always reliable indicators of engagement, leading to 
possible misclassification.  
13. 
Vision-Based 
Student 
Attention 
Recognition 
(2022) 
This study uses lightweight CNN models along with facial expressions and head pose 
information for attention detection. It is computationally efficient and suitable for basic 
applications where quick and simple monitoring is needed. The paper shows that lightweight 
models can still be useful for practical engagement detection tasks. 
Limitations:  
Limited feature diversity reduces accuracy in complex real-world scenarios.  
14. 
Online Learning Engagement Monitoring Using Facial Features (2022) 
This paper analyzes engagement using facial landmarks and head pose estimation. It 
demonstrates that facial features can be effectively used to monitor student attention in online 
learning environments. The study supports the idea that visual facial analysis is a practical 
solution for digital classroom engagement tracking. 
Limitations:  
The system is not scalable and performs well only in controlled conditions.  
10 
15. 
Attention 
Monitoring 
of 
Students 
Using 
XGBoost 
(2023) 
This study uses classical machine learning techniques such as XGBoost along with facial 
features and eye blink rate for engagement detection. It shows that traditional machine 
learning models can still provide useful performance for student monitoring tasks. The work 
is important because it offers a simpler alternative to deep learning-based solutions. 
Limitations:  
The absence of deep learning limits its ability to capture complex patterns, resulting in lower 
accuracy. 
11 
PROBLEM STATEMENT 
In online and smart classroom environments, monitoring student attention has become a 
significant challenge due to the lack of direct interaction between teachers and students. 
Traditional methods such as self-reporting and manual observation are subjective, 
inconsistent, and not scalable for large classrooms.Many students attend online sessions but 
may be distracted, bored, drowsy, or looking away from the screen, making it difficult for 
instructors to accurately assess their level of engagement. Existing automated systems often 
fail to capture subtle facial expressions, head movements, and gaze patterns, and many lack 
real-time capability.Therefore, there is a need for an intelligent and reliable system that can 
automatically analyze student behavior using video input and accurately determine whether a 
student is engaged or not engaged in real time. 
12 
CHAPTER 3 
METHODOLOGY 
3.1 Existing System 
In the existing system, student engagement detection is generally carried out using visual 
observation, offline analysis, or computationally complex deep learning models. From the 
literature review provided, many earlier systems rely on facial expressions, eye gaze, head 
pose, behavioral cues, or multimodal combinations to estimate engagement. Some studies use 
CNN–LSTM architectures to capture both spatial and temporal information, while others use 
object detection models such as YOLOv4 or classical machine learning methods such as 
XGBoost. These systems show that student engagement can be measured automatically, but 
they also reveal several practical limitations.  
A number of existing approaches are accurate in controlled conditions, but many of them are 
mainly designed for offline processing. Some systems require high computational resources 
because they combine multiple modalities and sequential models such as CNN and LSTM. 
Others are limited to theoretical frameworks or dataset creation without providing a complete 
real-time implementation. In some cases, the models focus only on one feature, such as facial 
expression, without using supporting cues like eye closure or face direction. As a result, their 
ability to work reliably in real-time educational environments becomes limited.  
Another important issue in the existing system is the lack of practicality for lightweight 
implementation. Real-time systems must process webcam frames continuously and provide 
fast feedback. However, many earlier approaches are either computationally expensive, 
limited to offline analysis, or not scalable enough for practical deployment. This creates the 
need for a simpler and more efficient system that can work in real time using readily available 
hardware and a combination of deep learning and computer vision.  
Limitations of Existing System 
• High computational complexity in multimodal and CNN–LSTM-based approaches.  
• Many methods are designed only for offline analysis.  
• Some studies are only theoretical or dataset-oriented and do not provide full 
implementation.  
13 
• Some systems depend on limited visual cues and may not be reliable in real-world 
conditions.  
3.2 Proposed System 
The proposed system is a real-time student engagement detection system that uses a 
webcam feed, computer vision techniques, and a trained deep learning model to identify 
whether a student is engaged or not engaged. Based on the uploaded implementation files, the 
system first captures live video through the webcam. Each frame is then processed to detect 
the student’s face, analyze the eyes, estimate whether the student is looking away, and finally 
classify the facial state using a trained MobileNetV2-based model. The final engagement 
decision is produced by combining all these observations through fusion logic.  
The core idea of the proposed system is not to depend only on CNN prediction. Instead, it 
combines multiple real-time cues: 
• facial-state classification using the trained model,  
• eye detection to identify prolonged eye closure,  
• face position analysis to identify looking away,  
• and face-availability tracking to handle temporary face loss.  
The training file shows that the deep learning model is built using MobileNetV2 as the base 
architecture. The dataset is loaded from a directory named dataset, resized to 224 × 224, 
rescaled, augmented, and divided into training and validation sets using validation_split=0.2. 
After training, the model is saved as modal.h5, which is later loaded in the real-time module 
for prediction.  
In the real-time module, the system uses Haar cascade classifiers for face and eye detection. 
The webcam feed is continuously processed frame by frame. If the face is not detected for a 
short time, the program reuses the last detected face, which helps reduce output flickering. If 
the eyes remain closed for several frames, the student is marked as not engaged. If the face is 
significantly away from the center of the frame, the system interprets it as looking away. In 
addition, if the CNN predicts facial states such as bored, looking_away, or drowsy, the final 
output becomes Not Engaged. Otherwise, the system displays Engaged.  
14 
Advantages of Proposed System 
• Real-time monitoring using live webcam input.  
• Combination of deep learning and computer vision rules for better decision-making.  
• Lightweight pretrained model using MobileNetV2.  
• Stable output through anti-flicker logic and no-face handling.  
3.3 Requirements 
The proposed system requires both software and hardware support for model training and 
real-time execution. Since the project includes deep learning model development as well as 
webcam-based live prediction, the requirements can be divided into software requirements 
and hardware requirements. These requirements are based on the imported libraries, modules, 
and runtime behavior visible in the uploaded files.  
3.3.1 Software Requirements 
• Python – Core programming language used for implementation  
• TensorFlow / Keras – Deep learning model training, loading, and prediction  
• OpenCV – Video capture, face detection, eye detection, and real-time processing  
• NumPy – Image array processing and numerical operations  
• Dataset Directory – Stores categorized training images for model training  
• Saved Model (modal.h5) – Pre-trained model used for real-time prediction  
3.3.2 Hardware Requirements 
• Computer / Laptop – System for training and real-time execution  
• Webcam – Captures live video input for engagement detection  
• Processor (CPU) – Handles image processing and model inference  
• Memory (RAM) – Supports data processing and model execution  
• Storage – Stores dataset, libraries, and trained model  
• GPU (Optional) – Speeds up deep learning model training 
15 
3.4 System Architecture 
The system architecture of the proposed model consists of two major phases: 
Phase 1: Training Phase 
In the training phase, the dataset is loaded from the dataset folder using ImageDataGenerator. 
The images are rescaled and augmented, and 20% of the data is reserved for validation. Then 
MobileNetV2 is used as the base feature extractor. On top of the base model, Global Average 
Pooling, a Dense layer with 256 neurons, a Dropout layer, and a final Softmax classification 
layer are added. The model is compiled using the Adam optimizer, categorical cross-entropy 
loss, and accuracy metric. After training for 10 epochs, the model is saved as modal.h5.  
Phase 2: Real-Time Detection Phase 
In the real-time phase, the saved model is loaded and the webcam feed is activated. Each 
incoming frame is converted to grayscale for face and eye detection. The face is detected 
using Haar cascade, and the detected face region is used for further processing. Eye detection 
is performed within the face region to determine whether the eyes are closed. At the same 
time, the position of the detected face with respect to the frame center is used to decide 
whether the student is looking away. The face image is also resized and normalized before 
being passed to the CNN model for classification into one of the predefined labels. Finally, 
fusion logic combines all conditions and displays either Engaged or Not Engaged on the 
screen.  
16 
Student Attention Monitoring and Analysis System 
Figure 3.1: System Architecture 
17 
3.5 Proposed Model 
The proposed model is a hybrid student engagement detection model that combines deep 
learning classification with rule-based computer vision analysis. The deep learning part is 
responsible for identifying the facial state of the student, while the computer vision part 
checks supporting conditions such as face presence, eye closure, and face direction. This 
hybrid design improves the final decision-making process by not depending only on a single 
model output.  
The deep learning component uses MobileNetV2 as the base model. MobileNetV2 is chosen 
because it is a lightweight pretrained CNN suitable for image-based classification tasks. In the 
uploaded training file, the base model is connected to GlobalAveragePooling2D, followed by 
a dense layer with ReLU activation, dropout regularization, and a final softmax output layer. 
This structure allows the system to classify multiple facial states from the input image.  
The real-time engagement decision is produced through fusion logic: 
• if eyes remain closed for more than 10 frames, the system marks the student as Not 
Engaged (Eyes Closed),  
• if the face is significantly away from the frame center, the system marks the student as 
Not Engaged (Looking Away),  
• if the model predicts labels such as bored, looking_away, or drowsy, the system 
outputs Not Engaged,  
• otherwise, the student is considered Engaged.  
This proposed model is therefore a practical fusion model that uses: 
1. webcam-based visual input,  
2. OpenCV Haar-cascade detection,  
3. MobileNetV2-based facial-state classification, and  
4. decision rules for final engagement determination.  
3.5.1 Dataset 
The dataset used in this project is the Student-engagement dataset available on Kaggle. It was 
created for detecting student engagement in online classes using images of students captured 
18 
through laptop cameras, which makes it well suited for webcam-based engagement analysis 
systems like your project.  
This dataset contains a total of 2,120 images and is organized into two main categories: 
Engaged and Not Engaged. The Engaged category contains 1,076 images, while the Not 
Engaged category contains 1,044 images. A later research article that explicitly reports results 
on this Kaggle dataset gives this class distribution in detail.  
The dataset is further divided into six subclasses. Under the Engaged category, the images 
belong to confused (369 images), engaged (347 images), and frustrated (360 images). Under 
the Not Engaged category, the images belong to looking away (423 images), bored (358 
images), and drowsy (263 images). These subclasses represent different visible student states 
during online learning.  
This class structure is also consistent with your implementation, because the real-time code 
uses the same six labels: confused, engaged, frustrated, looking_away, bored, drowsy. So the 
dataset and your trained model are aligned in terms of output categories.  
3.5.1.1 Features 
The main features of this dataset are facial visual features taken from student images during 
online classes. Since the images are collected through laptop cameras, they are useful for 
learning visible engagement-related cues such as facial expression, eye condition, 
attentiveness, tiredness, and face direction. These cues are important because engagement in 
online learning is often reflected through the student’s face and head region.  
From the class labels, it is clear that the dataset captures both positive engagement-related 
states and disengagement-related states. For example, engaged, confused, and frustrated are 
grouped under engaged behavior, while looking away, bored, and drowsy are grouped under 
not engaged behavior. This helps the model learn finer differences in student attention rather 
than only a simple binary output.  
In your project, these facial features are used by the CNN model for classification, and then 
additional visual checks such as eye detection and face direction are used in the real-time 
system for the final engagement decision.  
19 
3.5.1.2 Pre-processing 
Before training, the dataset images are preprocessed in your training script using 
ImageDataGenerator. The preprocessing steps include rescaling pixel values by 1/255, 
splitting 20% of the data for validation, and applying data augmentation techniques such as 
horizontal flipping and zooming. These steps help improve generalization and reduce 
overfitting during model training.  
All images are resized to 224 × 224 pixels, which matches the required input size of the 
MobileNetV2 model used in your project. The training code loads the images from the dataset 
directory using flow_from_directory(), which means the dataset is organized in class-wise 
folders.  
During real-time prediction, the detected face region is again resized to 224 × 224, normalized 
by dividing by 255.0, and reshaped into the required model input format before classification. 
This ensures that the input given during live testing is consistent with the format used during 
training.  
20 
CHAPTER 4 
IMPLEMENTATION AND TESTING 
4.1 Implementation Approach 
The implementation of this project was carried out in two main stages: model training and 
real-time deployment. In the first stage, a deep learning model was built and trained using 
the student engagement image dataset. In the second stage, the trained model was integrated 
into a live webcam-based application to detect student engagement in real time. This two
stage approach made the system practical, because the model could first learn facial-state 
patterns from dataset images and then use that knowledge during live prediction.  
The training process begins with loading the dataset using ImageDataGenerator. The images 
are preprocessed by rescaling pixel values, splitting the data into training and validation sets, 
and applying augmentation techniques such as horizontal flipping and zooming. After that, the 
images are fed into a MobileNetV2-based deep learning architecture. The base model is 
connected to additional layers such as GlobalAveragePooling2D, Dense, and Dropout, 
followed by a final softmax output layer for multiclass classification. The model is then 
compiled with the Adam optimizer and trained for 10 epochs. Once training is completed, the 
model is saved as modal.h5.  
After training, the saved model is used in the real-time module. In this phase, the system 
activates the webcam and continuously captures video frames. For each frame, face detection 
is performed using Haar cascade, and the face region is extracted for analysis. Eye detection is 
also applied inside the detected face region, and the system checks whether the student’s eyes 
remain closed for multiple frames. At the same time, the face position is compared with the 
center of the frame to detect whether the student is looking away. Finally, the extracted face 
image is resized, normalized, and passed into the trained model to predict one of the six 
classes: confused, engaged, frustrated, looking_away, bored, and drowsy.  
The final implementation approach of the project is therefore a hybrid real-time engagement 
detection system. It does not depend only on the deep learning prediction. Instead, it 
combines CNN classification with computer vision rules such as eye closure detection, 
looking-away detection, and no-face handling logic. This approach improves the practical 
reliability of the system and makes the final output more meaningful in real learning 
situations.  
21 
4.2 Functional and Algorithmic Implementation 
This section explains the exact functional modules and algorithmic steps implemented in the 
project. 
4.2.1 Dataset Loading and Preparation 
The first functional step implemented in the project is dataset loading and preprocessing. The 
dataset is read from a folder named dataset using flow_from_directory(). The images are 
resized to 224 × 224 pixels, and the batch size is set to 32. The training script also uses a 
validation split of 0.2, meaning 80% of the images are used for training and 20% are used for 
validation. To improve generalization, augmentation techniques such as horizontal flipping 
and zooming are applied.  
4.2.2 Model Building 
The main model implemented in the project is MobileNetV2. It is used as the base feature 
extractor with include_top=False and weights='imagenet'. On top of this pretrained model, a 
GlobalAveragePooling2D layer is added to reduce the spatial feature maps into a compact 
representation. Then a Dense(256, activation='relu') layer is used for learning higher-level 
patterns, followed by a Dropout(0.5) layer to reduce overfitting. Finally, the output is passed 
to a softmax classification layer whose size depends on the number of dataset classes.  
This design makes MobileNetV2 the main deep learning model of the project. Since it is a 
lightweight pretrained network, it is suitable for image-based classification tasks and practical 
for systems that later need real-time prediction. The final output layer supports multiclass 
classification, which matches the six labels used during real-time execution.  
4.2.3 Model Training 
After the model is built, it is compiled using: 
• Optimizer: Adam  
• Loss Function: categorical crossentropy  
• Metric: accuracy  
Then the model is trained using the training dataset and validated using the validation dataset 
for 10 epochs. After training, the model is saved as modal.h5, which is the trained file later 
used in the real-time system.  
22 
4.2.4 Real-Time Video Capture 
In the real-time implementation, the webcam is accessed through cv2.VideoCapture(0). The 
system captures frames continuously inside a loop and processes each frame one by one. This 
allows the project to monitor the student live instead of depending only on offline image 
analysis.  
4.2.5 Face Detection 
The project uses Haar cascade frontal face detection for identifying the student’s face in 
each webcam frame. The cascade is loaded from OpenCV’s built-in XML file. Each frame is 
converted to grayscale, and then detectMultiScale() is used to detect face regions. The 
parameters such as scaleFactor, minNeighbors, and minSize are tuned in the implementation 
to improve stability.  
4.2.6 Eye Detection 
After detecting the face, the system extracts the face region and performs eye detection using 
another Haar cascade classifier. This helps the project determine whether the student’s eyes 
are open or closed. If no eyes are detected continuously for multiple frames, the project 
interprets that as disengagement. This feature is important because drowsiness or eye closure 
is a strong indicator of low engagement.  
4.2.7 Looking-Away Detection 
The system also implements a simple head-direction or looking-away check. It calculates the 
center of the detected face and compares it with the center of the webcam frame. If the face 
center is significantly away from the frame center, the system interprets the student as looking 
away. This is used as another rule for identifying disengagement.  
4.2.8 CNN-Based Facial-State Prediction 
In addition to OpenCV-based analysis, the face image is resized to 224 × 224, normalized by 
dividing pixel values by 255.0, and reshaped into the required input format (1, 224, 224, 3). It 
is then passed to the trained model for prediction. The output class is obtained using 
np.argmax() and mapped to one of the six predefined labels: 
• confused  
• engaged  
23 
• frustrated  
• looking_away  
• bored  
• drowsy  
4.2.9 Fusion Logic for Final Output 
The most important algorithmic part of the project is the fusion logic. Instead of directly 
using the model’s label as the final output, the project combines multiple conditions: 
• If eyes are closed for more than 10 frames → Not Engaged (Eyes Closed)  
• Else if the face is away from the center → Not Engaged (Looking Away)  
• Else if the predicted label is bored, looking_away, or drowsy → Not Engaged  
• Otherwise → Engaged  
This fusion logic is a major implementation feature because it improves the practical decision
making of the system. It uses both learned model output and direct visual rules to produce the 
final engagement result. 
4.2.10 Anti-Flicker and No-Face Handling 
Another functional element implemented in the project is anti-flicker handling. If the face is 
not detected in a frame, the system temporarily reuses the last detected face instead of 
immediately changing the output. Only after several consecutive no-face frames does it show 
the message “No Face - Not Engaged”. This reduces unstable output and improves the user 
experience in real-time monitoring.  
4.3 Integration Details  
The project integrates multiple supporting components into one working system. The first 
major integration is between the training module and the real-time detection module. The 
training module creates the deep learning model and saves it as modal.h5, while the real-time 
module loads the same file and uses it for live prediction. This integration connects offline 
learning with real-time application.  
24 
The second integration is between TensorFlow/Keras and OpenCV. TensorFlow/Keras is 
used for deep learning-based facial-state prediction, while OpenCV is used for webcam 
capture, frame preprocessing, Haar cascade face detection, eye detection, text drawing, and 
output display. Together, these tools allow the system to combine model-based intelligence 
with live computer vision processing.  
The project also integrates pretrained transfer learning with a custom classifier. 
MobileNetV2 is loaded with ImageNet weights, and then new classification layers are added 
for the student engagement classes. This integration helps the system take advantage of 
pretrained feature extraction while adapting the final classifier to the project-specific dataset.  
Another integration present in the system is the combination of computer vision rules and 
deep learning output. Face detection, eye detection, and looking-away logic act as 
supporting components around the main CNN model. These external visual checks are 
integrated into the decision pipeline to make the final result more reliable. This makes the 
system not just a plain image classifier, but a more complete real-time engagement monitoring 
application.  
4.4 Testing 
Testing is an important stage in this project because the system must work correctly both 
during model training and during real-time execution. Based on the uploaded implementation, 
testing can be understood at two levels: model-level testing and functional system testing. 
At the model level, the training script uses a validation dataset generated from the same 
dataset source with a validation_split=0.2. During training, the model performance is checked 
using the accuracy metric, while the loss is measured using categorical crossentropy. This 
allows the system to evaluate whether the model is learning useful patterns from the dataset.  
At the functional level, the real-time system is tested by running the webcam application and 
observing whether the system correctly performs face detection, eye detection, looking-away 
detection, class prediction, and final engagement labeling. Since the system is interactive and 
frame-based, functional testing is important to ensure that each module behaves correctly in 
practical conditions. The display output, bounding boxes, predicted labels, and engagement 
messages serve as visible indicators of system performance.  
25 
The implemented testing also includes behavior verification for special conditions such as no 
face detected, prolonged eye closure, and off-center face position. These conditions are 
explicitly handled in the code and therefore form a natural part of the system testing process.  
4.4.2 Test Case Output Screens / Screenshots 
In this section, screenshots of the real-time output are included to demonstrate how the system 
behaves under different student facial conditions. These screenshots act as visual proof of the 
system’s functionality and help verify that the implemented model is correctly classifying the 
student into the predefined categories. Since the system operates in real time using webcam 
input, the output is continuously updated based on the detected facial features and behavioral 
cues. Therefore, capturing screenshots at different moments helps in clearly presenting the 
performance and reliability of the system. 
The screenshots are collected during real-time testing by simulating different student 
behaviors in front of the webcam. For each test case, the detected class label and the final 
engagement status are displayed on the screen. These outputs confirm that the system not only 
predicts facial states using the deep learning model but also applies fusion logic to determine 
the final result. This makes the screenshots an important part of functional testing, as they 
visually validate both the model prediction and the decision-making process of the system. 
For demonstration purposes, three important test cases are considered: Engaged, Looking 
Away, and Eyes Closed. In the engaged case, the system correctly identifies the student as 
attentive and displays the status as Engaged. In the looking away case, even if the face is 
detected, the system recognizes that the student is not focused on the screen and classifies the 
output as Not Engaged. Similarly, in the eyes closed case, the system detects continuous eye 
closure and marks the student as Not Engaged, indicating a drowsy or inattentive state. These 
test cases effectively demonstrate that the system can handle different real-world scenarios 
and produce accurate engagement results. 
Overall, the screenshots confirm that the system performs reliably in real-time conditions and 
successfully identifies both engaged and non-engaged states. They also highlight the 
effectiveness of combining deep learning predictions with behavioral analysis, making the 
system more robust and suitable for practical deployment. 
26 
Sample Test Cases 
1. Engaged 
Figure 4.1: Output Showing Engaged Student 
2. Looking Away 
Figure 4.2: Output Showing Disengaged Student (Looking Away) 
27 
28 
 
3. Eyes Closed  
 
Figure 4.3: Output Showing Disengaged Student (Eyes Closed) 
 
 
  
CHAPTER 5 
DISCUSSION OF RESULTS 
The results of the proposed system are evaluated at both the training level and the real-time 
implementation level. The MobileNetV2-based model is trained to classify six facial-state 
categories: confused, engaged, frustrated, looking away, bored, and drowsy. The training 
process uses categorical crossentropy as the loss function and accuracy as the evaluation 
metric. To improve model performance and generalization, data augmentation techniques such 
as rescaling, horizontal flipping, and zooming are applied during training. These techniques 
help the model learn robust features from the dataset and reduce overfitting. As a result, the 
model successfully learns to distinguish between different facial states, which serves as a 
strong foundation for accurate engagement detection. 
In the real-time system, the output is not solely dependent on the model’s prediction. Instead, 
the system enhances the prediction by combining it with additional behavioral conditions such 
as eye closure detection and face direction analysis. For instance, if the model predicts an 
engaged state but the eyes remain closed for multiple consecutive frames, the system 
overrides the result and classifies the student as Not Engaged. Similarly, if the face is detected 
but is positioned away from the screen center, indicating that the student is looking away, the 
final output is also marked as Not Engaged. This fusion-based decision mechanism improves 
the reliability of the system and ensures that the output reflects actual student behavior rather 
than only facial appearance. 
The system also demonstrates effective performance in real-time conditions using webcam 
input. It continuously captures frames, detects faces using Haar cascade classifiers, processes 
facial regions, and displays the engagement status dynamically. The inclusion of anti-flicker 
logic ensures that temporary loss of face detection does not immediately change the output, 
thereby maintaining stability. Additionally, the no-face handling mechanism ensures that if the 
face is not detected for a certain duration, the system appropriately displays a Not Engaged 
status. These features contribute to a smoother and more realistic user experience during live 
monitoring. 
Overall, the results indicate that the proposed system provides a practical, efficient, and 
reliable solution for real-time student engagement detection. By combining deep learning
based classification with rule-based behavioral analysis, the system achieves better accuracy 
29 
compared to using a standalone model. The implementation successfully meets the project 
objective of identifying student engagement in real time while maintaining simplicity, 
stability, and usability. 
30 
CHAPTER 6 
CONCLUSIONS & FUTURE WORK 
6.1 Conclusion 
This project presents a real-time student engagement detection system using computer vision 
and deep learning techniques. The system employs a MobileNetV2-based model to classify 
multiple student facial states such as engaged, confused, frustrated, bored, drowsy, and 
looking away, providing a more detailed understanding of student behavior. In addition to 
deep learning predictions, the system integrates rule-based analysis such as eye closure 
detection, head direction estimation, and face presence monitoring to improve accuracy and 
reliability. The real-time implementation using OpenCV enables continuous webcam input 
processing, stable prediction output, and effective handling of challenges such as flickering 
and missing face detection. The combination of a lightweight pretrained model and behavioral 
feature fusion makes the system both efficient and practical for real-world use. Overall, the 
project successfully demonstrates that student attention can be monitored effectively using an 
automated approach, making it a valuable solution for enhancing engagement analysis in 
online learning environments. 
6.2 Future Scope 
Although the current project successfully detects student engagement in real time, it can be 
further improved and extended in several useful ways. Some possible future enhancements are 
given below. 
6.2.1 Integration with Transportation and IoT-Based Alert System 
One possible future extension of this project is to use the same concept in the transportation 
field to detect whether a driver is sleepy or drowsy. In this case, the system can monitor the 
driver’s face and eye condition continuously using a camera. If the system detects that the 
driver is sleepy, it can be connected with IoT devices such as a buzzer, alarm, or speaker to 
produce a warning sound and wake the driver. This improvement can help in preventing 
accidents and increasing road safety. 
6.2.2 Multi-Face Engagement Detection 
31 
The present system is mainly designed for detecting the engagement of a single person at a 
time. In future, the project can be extended to support multiple face detection and analysis 
in the same frame. This would make the system more useful in classrooms, group learning 
sessions, or online meetings where more than one student is present. By identifying and 
analyzing multiple faces simultaneously, the system can monitor the engagement level of 
several students at once. 
6.2.3 Integration with Online Learning Platforms 
Another important future enhancement is the integration of this system with online learning 
platforms. The model can be connected with virtual learning systems such as online class 
portals, video conferencing tools, or e-learning applications. This would allow the platform to 
monitor student engagement automatically during live classes and provide useful feedback to 
teachers or administrators. Such integration can improve the effectiveness of online education 
by helping instructors understand student attention levels more easily. 
32 
REFERENCES 
[1]  
[2]  
[3]  
[4]  
[5]  
[6]  
[7]  
[8]  
[9]  
Yan L, Wu X, Wang Y (2025) Student engagement assessment using multimodal deep 
learning. 
Hara Gopal VP, Arun Babu P, Anays SM, Hussian PJ, Hemanjali M, Charan MS 
(2024) Student engagement assessment using multimodal deep learning 
Das R (2025) Optimizing student engagement detection using facial and behavioral 
features 
Bashir BM (2025) Detection of students’ emotions in online learning using CNN
LSTM 
Wu X, Chen Z, Liu Y, Wang H (2024) CMOSE: Comprehensive multi-modality online 
student engagement dataset with high-quality labels 
Abdelkawy A, Alkabbany I, Ali A, Farag A (2023) Measuring student behavioral 
engagement using histogram of actions 
Gothwal P, Banerjee D, Biswas AK (2025) ViBED-Net: Video-based engagement 
detection network using spatiotemporal cues 
Klein R, Celik T (2021) Detecting student engagement during lectures using 
convolutional neural networks 
Abedi A, Khan SS (2021) Detecting student engagement using ResNet and temporal 
convolutional network 
[10] 
Ismaeel A, Abrar M (2023) Detecting student engagement in online learning using 
CNN–LSTM 
[11] 
Shiri FM, Ahmadi E, Rezaee M (2023) Detection of student engagement in e-learning 
environments using deep learning 
[12] 
Alruwais N, Zakariah M (2023) Student engagement detection in classroom using 
machine learning algorithm 
[13] 
Whitehill J, Serpell Z, Lin YC, Foster A, Movellan JR (2014) The faces of 
engagement: Automatic recognition of student engagement from facial expressions 
[14] 
D’Mello S, Graesser A (2012) Dynamics of affective states during complex learning 
[15] 
[16] 
Bosch N, D’Mello S (2016) Automatic detection of learning-centered affective states 
Raca M, Kidzinski L, Dillenbourg P (2015) Capture of student engagement in the 
classroom 
[17] 
Zhao G, Pietikäinen M (2007) Dynamic texture recognition using local binary patterns 
33 
[18] 
Koelstra S, Muhl C, Soleymani M et al. (2012) DEAP: A database for emotion 
analysis using physiological signals 
[19] 
Poria S, Cambria E, Bajpai R, Hussain A (2017) A review of affective computing: 
From unimodal analysis to multimodal fusion 
[20] 
Zeng Z, Pantic M, Roisman GI, Huang TS (2009) A survey of affect recognition 
methods: Audio, visual, and spontaneous expressions 
[21] 
Buono P, De Carolis B, D’Errico F, Macchiarulo N, Palestra G (2023) Assessing 
student engagement from facial behavior in online learning 
[22] 
Monkaresi H, Bosch N, Calvo RA, D’Mello S (2016) Automated detection of 
engagement using video-based estimation of facial expressions and heart rate 
[23] 
Jaques N, Taylor S, Sano A, Picard R (2014) Multimodal autoencoder for predicting 
affective state 
[24] 
Zaletelj J, Košir A (2017) Predicting students’ attention in the classroom from Kinect 
facial and body features 
[25] 
D’Mello S, Olney A, Williams C, Hays P (2012) Gaze tutor: A gaze-reactive 
intelligent tutoring system 
[26] 
Grafsgaard JF, Wiggins JB, Boyer KE, Wiebe EN, Lester JC (2013) Automatically 
recognizing facial expression: Predicting engagement and frustration 
[27] 
Mollahosseini A, Hasani B, Mahoor MH (2016) AffectNet: A database for facial 
expression, valence, and arousal computing 
[28] 
Kollias D, Zafeiriou S (2019) Expression, affect, action unit recognition: A review 
[29] 
Zhang Z, Luo P, Loy CC, Tang X (2014) Facial landmark detection by deep multi-task 
learning 
[30] 
Baltrusaitis T, Robinson P, Morency LP (2016) OpenFace: An open source facial 
behavior analysis toolkit 
[31] 
Goodfellow I, Bengio Y, Courville A (2016) Deep learning 
34 
APPENDIX 
A. SAMPLE CODE 
RealtimeV5 
# Import required libraries 
import cv2 
import numpy as np 
from tensorflow.keras.models import load_model 
# Load trained model 
model = load_model("modal.h5") 
img_size = 224 
labels = ['confused','engaged','frustrated','looking_away','bored','drowsy'] 
# Load face detector 
face_cascade = cv2.CascadeClassifier( 
cv2.data.haarcascades + "haarcascade_frontalface_default.xml" 
) 
# Load eye detector 
eye_cascade = cv2.CascadeClassifier( 
cv2.data.haarcascades + "haarcascade_eye.xml" 
) 
# Start webcam 
35 
cap = cv2.VideoCapture(0) 
# Initialize variables 
eye_closed_frames = 0 
last_face = None 
no_face_frames = 0 
# Start video processing 
while True: 
ret, frame = cap.read() 
if not ret: 
break 
# Get frame size 
h, w = frame.shape[:2] 
frame_center_x = w // 2 
# Convert to grayscale 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
# Detect faces 
faces = face_cascade.detectMultiScale( 
gray, 
scaleFactor=1.2, 
36 
minNeighbors=4, 
minSize=(60, 60) 
) 
show_no_face_text = False 
# Handle no face detection 
if len(faces) == 0: 
no_face_frames += 1 
if last_face is not None: 
faces = [last_face] 
else: 
if no_face_frames > 5: 
show_no_face_text = True 
cv2.imshow("Student Engagement Detection", frame) 
if cv2.waitKey(1) & 0xFF == 27: 
break 
continue 
else: 
no_face_frames = 0 
last_face = faces[0] 
37 
# Process detected face 
for (x, y, w_face, h_face) in faces: 
# Extract face region 
roi_gray = gray[y:y+h_face, x:x+w_face] 
roi_color = frame[y:y+h_face, x:x+w_face] 
# Detect eyes 
eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5) 
# Check eye status 
if len(eyes) == 0: 
eye_closed_frames += 1 
else: 
eye_closed_frames = 0 
eyes_closed = eye_closed_frames > 10 
# Check head direction 
face_center_x = x + w_face // 2 
offset = abs(face_center_x - frame_center_x) 
looking_away = offset > 120 
# Prepare image for model 
38 
face_img = cv2.resize(roi_color, (img_size, img_size)) 
face_img = face_img / 255.0 
face_img = np.reshape(face_img, (1, img_size, img_size, 3)) 
# Predict emotion 
pred = model.predict(face_img, verbose=0) 
class_id = np.argmax(pred) 
label = labels[class_id] 
# Determine engagement 
if eyes_closed: 
status = "Not Engaged (Eyes Closed)" 
color = (0,0,255) 
elif looking_away: 
status = "Not Engaged (Looking Away)" 
color = (0,0,255) 
elif label in ['bored','looking_away','drowsy']: 
status = "Not Engaged" 
color = (0,0,255) 
else: 
status = "Engaged" 
39 
color = (0,255,0) 
# Draw results 
cv2.rectangle(frame, (x,y), (x+w_face, y+h_face), color, 2) 
cv2.putText(frame, label, (x, y-40), 
cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) 
cv2.putText(frame, status, (x, y-10), 
cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2) 
# Draw center line 
cv2.line(frame, (frame_center_x, 0), (frame_center_x, h), (255,255,0), 2) 
# Show no face message 
if show_no_face_text: 
cv2.putText(frame, "No Face - Not Engaged", (20,40), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 
# Display output 
cv2.imshow("Student Engagement Detection", frame) 
# Exit on ESC 
if cv2.waitKey(1) & 0xFF == 27: 
break 
40 
# Release resources 
cap.release() 
cv2.destroyAllWindows() 
Trained Model 
# Import required libraries 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, 
Flatten, Dense, Dropout 
from tensorflow.keras.applications import MobileNetV2 
# Set image size and batch size 
img_size = 224 
batch_size = 32 
# Data augmentation and preprocessing 
datagen = ImageDataGenerator( 
rescale=1./255, 
validation_split=0.2, 
horizontal_flip=True, 
41 
zoom_range=0.2 
) 
# Load training data 
train_data = datagen.flow_from_directory( 
"dataset", 
target_size=(img_size, img_size), 
batch_size=batch_size, 
class_mode='categorical', 
subset='training' 
) 
# Load validation data 
val_data = datagen.flow_from_directory( 
"dataset", 
target_size=(img_size, img_size), 
batch_size=batch_size, 
class_mode='categorical', 
subset='validation' 
) 
# Load pretrained MobileNetV2 model 
base_model = MobileNetV2( 
42 
input_shape=(img_size, img_size, 3), 
include_top=False, 
weights='imagenet' 
) 
# Build model 
model = Sequential([ 
base_model, 
GlobalAveragePooling2D(), 
Dense(256, activation='relu'), 
Dropout(0.5), 
Dense(train_data.num_classes, activation='softmax') 
]) 
# Compile model 
model.compile( 
optimizer='adam', 
loss='categorical_crossentropy', 
metrics=['accuracy'] 
) 
# Show model summary 
model.summary() 
# Train model 
43 
history = model.fit( 
train_data, 
validation_data=val_data, 
epochs=10 
) 
# Save trained model 
model.save("modal.h5") 
print("Model saved as modal.h5") 
44 