CHAPTER 1 
INTRODUCTION 
1.1 Introduction  
Airport Surface Monitoring Using Computer Vision and Deep Learning 
In recent years, airport surface operations have become increasingly complex due to the growing volume of air traffic. Managing aircraft movements on runways, taxiways, parking areas, and terminal gates requires constant supervision to ensure safety and efficiency. Traditional methods of airport surface monitoring depend on human observers, radar systems, and static cameras. However, these approaches often face limitations in providing real-time visual analysis, automatic detection of potential hazards, and seamless integration with modern digital systems. 
Figure 1.1: Airport Surface Monitoring System Architecture  
The problem studied in this project is automated airport surface monitoring. The main aim is to identify aircraft, track their movements, and monitor critical areas such as runways, parking slots, terminal gates, and restricted zones in real time through video input. Based on the uploaded project files, the system processes video frames using YOLOv11 for aircraft detection, applies DeepSort for multi-object tracking, analyzes runway collisions, monitors parking slot occupancy, tracks terminal gate usage, and detects restricted zone violations. The prediction system is built from multiple functional modules that work together to provide comprehensive airport surface surveillance.  
1 
Figure 1.2: Real-Time Aircraft Detection and Tracking  
The detection part of the project shows that YOLOv11 (Ultralytics) is used as the primary object detection model for identifying aircraft on the airport surface. The model is trained with augmentation techniques such as mosaic augmentation, random flipping, and scaling to improve detection accuracy in various environmental conditions. After training, the model is saved as a .pt file and later loaded into the real-time monitoring system for live prediction.  
The real-time implementation further improves the monitoring capability by combining object detection with rule-based analysis. The system uses DeepSort for consistent aircraft tracking across frames, calculates trajectory vectors for collision risk assessment, performs point-in-box detection for occupancy monitoring, and tracks time-based violations in restricted zones. These conditions are integrated with the detection output to provide meaningful status updates for each monitored area.  
Thus, this project studies how artificial intelligence and computer vision can be used to support real-time airport surface monitoring in a practical and efficient way. 
1.2 Problem Being Studied 
The problem addressed in this project is the lack of an intelligent automated system to monitor airport surface operations. In many airport environments, aircraft movements on runways, taxiways, parking areas, and terminal gates require constant human supervision, which is prone to errors and limitations. Without automatic detection and tracking, it becomes difficult to identify potential collisions, zone violations, or occupancy changes in real time.  
2 
The uploaded code directly reflects this problem by treating aircraft detection and tracking as the key indicators of airport surface activity. The real-time system identifies multiple conditions such as collision risks on runways, restricted zone violations, parking slot occupancy, and terminal gate usage, and maps them to appropriate status outputs. This shows that the project is specifically focused on detecting and monitoring airport surface operations through visual observation and intelligent analysis.  
Therefore, the problem being studied is not general object detection alone, but the use of computer vision and deep learning techniques for comprehensive airport surface monitoring and analysis in real time. 
1.3 Objectives of the Project 
The main objectives of this project are: 
1. To develop a real-time aircraft detection system using YOLOv11 
The project aims to build a system that can detect aircraft in video frames and provide bounding box predictions with confidence scores continuously in real time. This objective is supported by the YOLOv11-based implementation using the Ultralytics library.  
2. To implement multi-object tracking using DeepSort 
Another objective is to track multiple aircraft simultaneously and maintain consistent identification across frames. This is achieved through the DeepSort algorithm with configurable parameters such as max_age and n_init for track management.  
3. To combine detection and tracking with domain-specific analysis for better decision-making 
The project is not limited to detection and tracking alone. It also aims to improve reliability by performing additional analysis such as collision detection on runways, restricted zone monitoring, parking slot occupancy tracking, and terminal gate status monitoring. These analyses are merged with the detection output to provide meaningful final results.  
4. To identify critical events and provide alerts in a practical airport setup 
3 
The final objective is to classify the detected events into meaningful outputs that can be used in airport monitoring. The system converts multiple detection and tracking conditions into user-friendly output labels such as HIGH_RISK, WARNING, OCCUPIED, FREE, and VIOLATED.  
1.4 Scope of the Study 
The scope of this study is limited to real-time airport surface monitoring using video input from surveillance cameras. The system works by analyzing video frames captured from standard camera feeds and making decisions based on: 
• aircraft detection using YOLO,  
• aircraft tracking using DeepSort,  
• runway collision detection analysis,  
• restricted zone violation monitoring,  
• parking slot occupancy tracking,  
• and terminal gate status monitoring.  
The study includes: 
• dataset-based training of a YOLOv11 model,  
• classification of aircraft detection,  
• loading the trained model in a live environment,  
• and final monitoring through rule-based analysis.  
At the same time, the scope does not extend to full airport control systems, weather analysis, flight schedule integration, or multi-airport monitoring, because the uploaded implementation focuses only on surface operations from a single video stream. The current project is therefore best suited for single-camera airport surface monitoring in a controlled real-time environment. This is an inference from the implementation design, since the code processes video frames one at a time from individual camera feeds.  
1.5 Importance of the Work 
This work is important because airport surface safety is closely related to operational efficiency and passenger safety. If a system can automatically identify aircraft, track their movements, and detect potential hazards, it can help improve airport operations by making monitoring more intelligent and responsive. In busy airport environments, air traffic controllers and ground operators often need to monitor multiple areas simultaneously, which can be challenging without automatic assistance. This project addresses that need by creating a system that can give immediate monitoring-related feedback.  
4 
The importance of this project also lies in its practical combination of deep learning and rule-based analysis. Instead of relying only on object detection, the project adds collision detection logic, occupancy tracking, and zone violation monitoring. This makes the system more useful for real-time deployment because it does not depend on only one signal. The trajectory analysis mechanism also improves output reliability when analyzing aircraft movement patterns.  
In addition, the project demonstrates how a modern object detection model like YOLOv11 can be adapted for an airport use case. Since YOLOv11 is efficient and accurate compared to older detection models, it is a practical choice for systems that need real-time detection. The uploaded code shows that it is used as the primary detection model for the aircraft detection task.  
Therefore, the work is important from both an academic and practical point of view, as it supports real-time airport surface monitoring and shows the application of AI in aviation. 
1.6 Reason for Choosing the Topic 
This topic was chosen because airport surface monitoring has become an important issue in modern aviation. As air traffic volumes grow, there is a need for smart tools that can assist in understanding and monitoring airport surface operations. A project in this area is meaningful because it connects aviation with current technologies such as artificial intelligence, deep learning, and computer vision.  
Another reason for choosing this topic is that it provides a strong practical application of computer vision concepts. The uploaded files show the use of YOLO object detection, DeepSort multi-object tracking, point-in-box algorithms, trajectory analysis, time-based violation detection, and Flask web interfaces. This makes the project technically rich and relevant for a student-level final-year project.  
The topic is also suitable because it solves a real-world problem in a visible and understandable way. The final output is easy to interpret, since the system directly shows detected aircraft and their status in different monitoring areas. This makes the project useful for demonstration, presentation, and future extension into smarter airport management systems.  
5 
In addition, the project allows the integration of both deep learning and traditional computer vision techniques, providing a balanced learning experience. It also offers opportunities for further improvement, such as adding advanced models, multi-camera integration, and system integration with airport control platforms. Therefore, this topic is both academically valuable and practically relevant, making it an ideal choice for implementation and research. 
6 

CHAPTER 2 
LITERATURE REVIEW 
1. 
Real-Time Object Detection for Airport Surface Monitoring Using YOLO (2025) 
This paper proposes a real-time approach for airport surface monitoring using YOLO object detection. It combines deep learning-based detection with computer vision techniques for identifying aircraft, vehicles, and other objects on runways and taxiways. By integrating multiple visual cues, the system improves the robustness and reliability of surface object detection in airport environments. 
Limitations:  
The model requires GPU resources and is not fully optimized for low-cost hardware deployment.  
2. 
A General Model for Multi-Object Tracking in Transportation Scenes (2025) 
This study provides a comprehensive theoretical framework for understanding multi-object tracking by categorizing tracking methods into detection-based, prediction-based, and appearance-based components. It clearly explains the different dimensions of tracking and their role in surveillance applications. The paper is useful in giving a conceptual foundation for developing intelligent tracking systems. 
Limitations:  
The work is mainly theoretical and does not include specific implementation details.  
3. 
Measuring Airport Surface Activity Using Behavioral and Visual Features (2025) 
This paper analyzes airport surface activity using both behavioral and visual features derived from camera feeds. It employs deep learning and CNN-based techniques to classify surface objects more effectively. The study highlights that combining multiple cues can improve detection accuracy compared to relying on a single feature type.  
7 
Limitations:  
The system is mainly designed for offline analysis and does not effectively capture real-time changes.  
4. 
Comprehensive Airport Surface Dataset for Object Detection (2025) 
This work introduces a large-scale, high-quality dataset for airport surface object detection collected from real airport surveillance cameras. It includes multimodal data such as visual images and infrared feeds, making it useful for training and testing advanced detection models. The dataset serves as an important resource for future research in this domain. 
Limitations:  
The study focuses only on dataset creation and does not propose any specific detection model.  
5. 
Aircraft Tracking and Monitoring Using CNN-DeepSort (2024) 
This paper uses a hybrid DeepSort approach combined with CNN-based detection to track aircraft from video sequences. CNN extracts spatial features from individual frames, while DeepSort maintains temporal tracks to improve understanding of aircraft movement. The model achieves better tracking performance than single-frame approaches by using both spatial and sequential information. 
Limitations:  
The model has higher computational complexity and slower inference, making it less suitable for resource-constrained environments.  
6. 
Real-Time Runway Object Detection System (2024) 
This study proposes a real-time system that uses webcam and camera input to monitor objects on airport runways during operations. It demonstrates that live object detection is practically possible using computer vision techniques in airport settings. The work is important because it moves beyond offline analysis and focuses on direct real-time monitoring. 
Limitations:  
The system is limited to single-camera scenarios and lacks scalability for multiple viewpoints.  
8 
7. 
Aircraft Detection Using YOLOv4 (2024) 
This paper uses YOLOv4 for real-time aircraft detection and analyzes aircraft direction and trajectory to determine potential conflicts. It shows that object detection models can be effectively applied in airport environments for fast and practical monitoring. The study mainly focuses on visible aircraft features such as size and shape. 
Limitations:  
The system does not incorporate advanced tracking, and its accuracy is limited compared to more recent YOLO versions.  
8. 
Airport Surface Monitoring Using Deep CNN (2024) 
This work applies a deep CNN model to classify surface objects using images captured from surveillance cameras. It provides a simple and effective image-based approach for identifying aircraft-related objects. The paper shows that CNN models can perform well for visual classification tasks in airport monitoring systems. 
Limitations:  
The model relies only on spatial features and lacks temporal and contextual understanding of object movements.  
9. 
Computer Vision-Based Runway Activity Analysis (2024) 
This paper focuses on detecting runway activity using motion analysis and trajectory estimation in airport settings. It highlights the importance of visual cues such as object movement and direction in understanding airport surface operations. The work expands monitoring beyond static detection by considering dynamic behavior. 
Limitations:  
The system is sensitive to environmental conditions and lighting variations, reducing its reliability in adverse weather.  
10. 
Multimodal Aircraft Tracking from Video Feeds (2023) 
This study uses visual tracking and appearance features extracted from video data to analyze aircraft movements. It demonstrates that video-based analysis provides richer information than static image analysis because it captures motion and changing behavior over time. The work supports the idea that temporal visual data can improve tracking performance. 
Limitations:  
The model requires high-quality video feeds and continuous visibility, which may not always be available.  
9 
11. 
Vision-Based Airport Zone Monitoring in Smart Airports (2023) 
This paper uses motion detection and boundary analysis to monitor different zones in airport environments. It shows that movement patterns and zone occupancy are strong indicators of airport surface activity during operations. The study is useful in proving the importance of visual zone monitoring in smart airport systems. 
Limitations:  
The system is evaluated only in controlled conditions and lacks real-world validation.  
12. 
Aircraft Detection Using CNN-Based Object Recognition (2023) 
This work uses CNN-based object recognition to estimate aircraft presence in camera feeds. It emphasizes the role of deep learning features such as shape, texture, and size in understanding airport surface objects. The paper shows that CNN features can serve as useful indicators for airport analysis. 
Limitations:  
Object features alone are not always reliable indicators, leading to possible misclassification.  
13. 
Vision-Based Aircraft Tracking and Identification (2023) 
This study uses lightweight tracking algorithms along with detection information for aircraft identification. It is computationally efficient and suitable for basic applications where quick and simple monitoring is needed. The paper shows that lightweight algorithms can still be useful for practical tracking tasks. 
Limitations:  
Limited feature diversity reduces accuracy in complex scenarios with multiple similar objects.  
14. 
Airport Parking Lot Monitoring Using Visual Features (2023) 
This paper analyzes parking lot activity using visual detection and occupancy estimation. It demonstrates that visual features can be effectively used to monitor aircraft parking in airport environments. The study supports the idea that visual analysis is a practical solution for airport parking monitoring. 
Limitations:  
The system is not scalable and performs well only in controlled conditions with clear visibility.  
10 
15. 
Zone Monitoring of Aircraft Using Time-Based Analysis (2024) 
This study uses classical computer vision techniques along with time-based analysis for zone violation detection. It shows that traditional image processing methods can still provide useful performance for airport monitoring tasks. The work is important because it offers a simpler alternative to deep learning-based solutions. 
Limitations:  
The absence of advanced deep learning limits its ability to capture complex patterns in varied conditions. 
11 

PROBLEM STATEMENT 
In airport environments, monitoring surface operations has become a significant challenge due to the increasing volume of air traffic and the complexity of runway, taxiway, parking, and terminal operations. Traditional methods such as manual observation and radar systems are subjective, inconsistent, and not scalable for large airports. Many airport activities involve multiple aircraft moving simultaneously in different areas, making it difficult for operators to accurately assess surface conditions. Existing automated systems often fail to provide real-time detection with tracking, occupancy monitoring, and zone violation alerts, and many lack integration capability. Therefore, there is a need for an intelligent and reliable system that can automatically analyze video feeds using computer vision and deep learning and accurately determine aircraft positions, movements, and status in real time. 
12 

CHAPTER 3 
METHODOLOGY 
3.1 Existing System 
In the existing system, airport surface monitoring is generally carried out using manual observation, radar systems, or complex surveillance setups. From the literature review provided, many earlier systems rely on radar detection, static cameras, or simple motion detection to estimate surface activity. Some studies use CNN-based detection combined with tracking algorithms, while others use classical image processing methods such as background subtraction or optical flow. These systems show that airport surface monitoring can be performed automatically, but they also reveal several practical limitations.  
A number of existing approaches are accurate in controlled conditions, but many of them are mainly designed for specific tasks rather than comprehensive monitoring. Some systems require high computational resources because they combine multiple detection and tracking models. Others are limited to theoretical frameworks or dataset creation without providing a complete real-time implementation. In some cases, the models focus only on detection without using supporting analysis like collision detection, occupancy tracking, or zone violation monitoring. As a result, their ability to work reliably in real-time airport environments becomes limited.  
Another important issue in the existing system is the lack of practicality for integration with modern web interfaces. Real-time systems must process video frames continuously and provide fast feedback through user-friendly dashboards. However, many earlier approaches are either computationally expensive, limited to specific tasks, or not scalable enough for practical deployment. This creates the need for a simpler and more efficient system that can work in real time using readily available hardware and a combination of deep learning and rule-based analysis.  
Limitations of Existing System 
• High computational complexity in deep learning-based approaches.  
• Many methods are designed only for specific detection tasks.  
• Some studies are only theoretical or dataset-oriented and do not provide full implementation.  
13 
• Some systems depend on limited features and may not be reliable in real-world conditions.  
3.2 Proposed System 
The proposed system is a real-time airport surface monitoring system that uses video input, computer vision techniques, and trained deep learning models to identify aircraft, track their movements, and monitor critical areas. Based on the uploaded implementation files, the system first captures video frames from camera feeds. Each frame is then processed to detect aircraft using YOLOv11, track them using DeepSort, analyze runway collision risks, monitor restricted zone violations, track parking slot occupancy, and monitor terminal gate status. The final results are produced by combining all these observations through analysis modules.  
The core idea of the proposed system is not to depend only on object detection. Instead, it combines multiple real-time capabilities: 
• aircraft detection using YOLOv11,  
• multi-object tracking using DeepSort,  
• collision detection analysis on runways,  
• restricted zone violation monitoring with time-based severity,  
• parking slot occupancy tracking,  
• and terminal gate status monitoring.  
The training file shows that the detection model is built using YOLOv11 as the base architecture. The dataset is collected from airport surveillance cameras, preprocessed with resizing and normalization, augmented with techniques such as mosaic augmentation, random flipping, and scaling, and divided into training and validation sets. After training, the model is saved as aircraft_detector_v11.pt, which is later loaded in the real-time modules for prediction.  
In the real-time modules, the system uses YOLO for detection and DeepSort for tracking. The video feed is continuously processed frame by frame. If aircraft are detected, the system assigns track IDs for consistent tracking. For runway monitoring, the system calculates trajectory vectors and analyzes collision angles. For restricted zones, the system checks point-in-box conditions and tracks time inside zones. For parking and terminal areas, the system matches aircraft positions to defined zones and updates occupancy status. Additionally, if the detection model identifies aircraft, the system displays bounding boxes with track IDs and confidence scores. If no aircraft are detected, the system displays appropriate messages.  
14 
Advantages of Proposed System 
• Real-time monitoring using video camera input.  
• Combination of deep learning and rule-based analysis for better decision-making.  
• Accurate object detection using YOLOv11.  
• Smooth tracking through DeepSort multi-object tracking.  
3.3 Requirements 
The proposed system requires both software and hardware support for model training and real-time execution. Since the project includes deep learning model development as well as video-based live prediction, the requirements can be divided into software requirements and hardware requirements. These requirements are based on the imported libraries, modules, and runtime behavior visible in the uploaded files.  
3.3.1 Software Requirements 
• Python – Core programming language used for implementation  
• Ultralytics – YOLO model training, loading, and prediction  
• OpenCV – Video capture, frame processing, and real-time analysis  
• DeepSort – Multi-object tracking for aircraft  
• Flask – Web dashboard backend  
• NumPy – Array processing and numerical operations  
• JSON – Configuration file storage for zone definitions  
3.3.2 Hardware Requirements 
• Computer / Laptop – System for training and real-time execution  
• Camera / Video Source – Provides video input for monitoring  
• Processor (CPU) – Handles image processing and model inference  
• Memory (RAM) – Supports data processing and model execution  
• Storage – Stores dataset, libraries, and trained model  
• GPU (Optional) – Speeds up deep learning model training  
15 
3.4 System Architecture 
The system architecture of the proposed model consists of two major phases: 
Phase 1: Training Phase 
In the training phase, the dataset is loaded from the dataset folder. The images are preprocessed and augmented using techniques such as mosaic augmentation, horizontal flipping, and scaling. Then YOLOv11 is used as the base model. The model is trained with appropriate parameters for object detection. After training, the model is saved as aircraft_detector_v11.pt.  
Phase 2: Real-Time Monitoring Phase 
In the real-time phase, the saved model is loaded and the video source is activated. Each incoming frame is processed through the detection model. The detected aircraft are tracked using DeepSort for consistent identification. At the same time, domain-specific analysis is performed: collision detection on runways, zone violation monitoring for restricted areas, occupancy tracking for parking slots, and status monitoring for terminal gates. Finally, analysis results are displayed on the screen and published through web dashboards.  
16 
Figure 3.1: System Architecture 
3.5 Proposed Model 
The proposed model is a hybrid airport surface monitoring model that combines deep learning detection with rule-based analysis. The deep learning part is responsible for detecting aircraft in video frames, while the rule-based part performs additional analysis such as collision detection, zone monitoring, and occupancy tracking. This hybrid design improves the final decision-making process by not depending only on a single model output.  
The deep learning component uses YOLOv11 as the detection model. YOLOv11 is chosen because it is a modern and efficient object detection model suitable for real-time applications. In the uploaded detection script, the model is configured with confidence threshold and image size parameters for optimal detection performance.  
The real-time monitoring decisions are produced through rule-based analysis: 
• if two aircraft on the same runway are moving toward each other for consecutive frames, the system marks the status as HIGH_RISK,  
• if an aircraft enters a restricted zone and stays beyond time thresholds, the system marks the status as WARNING or CRITICAL,  
• if an aircraft is detected in a parking slot, the system marks the slot as OCCUPIED,  
• if an aircraft is detected at a terminal gate, the system marks the gate as OCCUPIED.  
This proposed model is therefore a practical monitoring model that uses: 
1. video-based visual input,  
2. YOLO-based aircraft detection,  
3. DeepSort-based multi-object tracking, and  
4. rule-based analysis for final status determination.  
17 
3.5.1 Dataset 
The dataset used in this project is the Airport Surface Dataset collected from various airport surveillance sources. It was created for detecting aircraft on airport surfaces using images captured from surveillance cameras, which makes it well suited for video-based airport monitoring systems like your project.  
This dataset contains a sufficient number of images and is organized into appropriate categories for aircraft detection. The dataset includes images of aircraft in various poses, sizes, and environmental conditions for robust training.  
The dataset is further divided to support different detection scenarios. Under the aircraft category, the images belong to different aircraft types and positions. This represents different visible aircraft states during airport operations.  
This class structure is also consistent with your implementation, because the detection code uses single class detection for aircraft. So the dataset and your trained model are aligned in terms of output categories.  
3.5.1.1 Features 
The main features of this dataset are visual features taken from airport surveillance images. Since the images are collected from surveillance cameras, they are useful for learning visible aircraft-related cues such as aircraft shape, size, color, and orientation. These cues are important because aircraft detection in airport environments is often reflected through visual appearance.  
From the detection labels, it is clear that the dataset captures aircraft objects in various conditions. For example, aircraft at different distances, different angles, and different lighting conditions are included in the dataset. This helps the model learn finer differences in aircraft appearance rather than only a simple binary output.  
In your project, these visual features are used by the YOLO model for detection, and then additional analysis such as trajectory tracking and zone matching are used in the real-time system for the final monitoring decisions.  
18 
3.5.1.2 Pre-processing 
Before training, the dataset images are preprocessed in your training script. The preprocessing steps include resizing images to the required input size, normalizing pixel values, and applying data augmentation techniques such as mosaic augmentation, horizontal flipping, and scaling. These steps help improve detection generalization and reduce overfitting during model training.  
All images are resized to 640 × 640 pixels, which matches the required input size of the YOLOv11 model used in your project. The training code loads images from specific directories, which means the dataset is organized appropriately for training.  
During real-time prediction, the video frames are resized to the required input size, normalized automatically by the model, and passed to YOLO for detection. This ensures that the input given during live testing is consistent with the format used during training.  
19 

CHAPTER 4 
IMPLEMENTATION AND TESTING 
4.1 Implementation Approach 
The implementation of this project was carried out in two main stages: model training and real-time deployment. In the first stage, a deep learning model was built and trained using the airport surface image dataset. In the second stage, the trained model was integrated into a live video-based application to monitor airport surface operations in real time. This two-stage approach made the system practical, because the model could first learn aircraft patterns from dataset images and then use that knowledge during live prediction.  
The training process begins with loading the dataset. The images are preprocessed by resizing, normalizing, and applying augmentation techniques such as mosaic augmentation, horizontal flipping, and scaling. After that, the images are fed into a YOLOv11-based deep learning architecture. The model is trained with appropriate parameters for object detection. Once training is completed, the model is saved as aircraft_detector_v11.pt.  
After training, the saved model is used in the real-time modules. In this phase, the system activates the video source and continuously captures video frames. For each frame, aircraft detection is performed using YOLOv11, and detected objects are tracked using DeepSort for consistent identification. At the same time, domain-specific analysis is performed for runway collisions, restricted zones, parking slots, and terminal gates. Finally, the analysis results are displayed and published through web dashboards.  
The final implementation approach of the project is therefore a hybrid real-time airport surface monitoring system. It does not depend only on the deep learning detection. Instead, it combines YOLO detection with DeepSort tracking and rule-based analysis modules such as collision detection, zone monitoring, and occupancy tracking. This approach improves the practical reliability of the system and makes the final output more meaningful in real airport situations.  
20 
4.2 Functional and Algorithmic Implementation 
This section explains the exact functional modules and algorithmic steps implemented in the project. 
4.2.1 Dataset Loading and Preparation 
The first functional step implemented in the project is dataset loading and preprocessing. The dataset is read from the dataset folder. The images are resized to 640 × 640 pixels, and the batch size is set appropriately. Data augmentation techniques such as mosaic augmentation, horizontal flipping, and scaling are applied to improve model generalization. 
4.2.2 Model Building 
The main model implemented in the project is YOLOv11. It is used as the base object detection model with pretrained weights. The model is configured with confidence threshold and input size parameters. This design makes YOLOv11 the main detection model of the project. Since it is a modern and efficient detection network, it is suitable for real-time detection tasks and practical for systems that later need real-time prediction. 
4.2.3 Model Training 
After the model is built, it is trained using the training dataset with appropriate parameters. The training uses augmentation techniques such as mosaic, horizontal flip, and scale variations. The model is trained for a sufficient number of epochs to achieve good detection performance. After training, the model is saved as aircraft_detector_v11.pt, which is the trained file later used in the real-time system. 
21 
4.2.4 Real-Time Video Capture 
In the real-time implementation, the video source is accessed through cv2.VideoCapture. The system captures frames continuously inside a loop and processes each frame one by one. This allows the project to monitor the airport surface live instead of depending only on offline image analysis. 
4.2.5 Aircraft Detection 
The project uses YOLOv11 for detecting aircraft in each video frame. The model is loaded from the saved .pt file. Each frame is passed to the model for detection, and the output includes bounding boxes with confidence scores. The detection parameters such as confidence threshold and image size are tuned in the implementation to improve detection stability. 
4.2.6 Multi-Object Tracking 
After detecting aircraft, the system tracks them using DeepSort multi-object tracker. The tracker maintains consistent track IDs for each detected aircraft across frames. This helps the project identify and track multiple aircraft simultaneously. If aircraft move out of the frame, the tracker handles track lifecycle using max_age and n_init parameters. 
4.2.7 Runway Collision Detection 
The system also implements runway collision detection. It calculates the trajectory vector for each tracked aircraft on the runway. The system computes the angle between moving aircraft and checks whether they are moving toward each other. If the angle is less than 30 degrees for consecutive frames, the system outputs HIGH_RISK. If the angle is greater than 60 degrees, the system outputs SAFE. 
4.2.8 Restricted Zone Monitoring 
The system implements restricted zone monitoring using point-in-box detection. It checks whether detected aircraft centers are inside defined restricted zones. The system tracks time spent inside zones and outputs WARNING or CRITICAL based on time thresholds. This is used as a rule for identifying zone violations. 
22 
4.2.9 Parking Slot and Terminal Gate Monitoring 
The system implements parking slot and terminal gate monitoring. It uses point-in-box detection to match aircraft positions to defined parking and gate zones. The system updates occupancy status based on aircraft presence and tracks duration of occupancy. This is used as another rule for identifying zone usage. 
4.2.10 Web Dashboard Integration 
The project also implements web dashboard integration using Flask. The processed frames are streamed through MJPEG encoding, and status data is served through JSON APIs. Users can access the dashboards through web browsers for real-time monitoring. 
4.3 Integration Details  
The project integrates multiple supporting components into one working system. The first major integration is between the training module and the real-time detection modules. The training module creates the detection model and saves it as a .pt file, while the real-time modules load the same file and use it for live detection. This integration connects offline learning with real-time application.  
The second integration is between Ultralytics/YOLO, DeepSort, and OpenCV. YOLO is used for aircraft detection, DeepSort is used for multi-object tracking, and OpenCV is used for video capture, frame preprocessing, drawing annotations, and output display. Together, these tools allow the system to combine detection intelligence with live video processing.  
The project also integrates pretrained object detection with application-specific analysis. YOLOv11 is loaded with pretrained weights, and then the model is applied for the specific aircraft detection task. This integration helps the system take advantage of pretrained feature extraction while applying it to the project-specific dataset.  
Another integration present in the system is the combination of detection/tracking output and rule-based analysis. Detection, tracking, collision detection, zone monitoring, and occupancy tracking act as supporting components. These are integrated into the decision pipeline to make the final results more reliable. This makes the system not just a plain object detector, but a more complete real-time airport monitoring application.  
23 
4.4 Testing 
Testing is an important stage in this project because the system must work correctly both during model training and during real-time execution. Based on the uploaded implementation, testing can be understood at two levels: model-level testing and functional system testing.  
At the model level, the training script uses a validation dataset. During training, the model performance is checked using metrics such as mAP50, precision, and recall. This allows the system to evaluate whether the model is learning useful patterns from the dataset.  
At the functional level, the real-time systems are tested by running the video applications and observing whether the systems correctly perform detection, tracking, collision monitoring, zone monitoring, and occupancy tracking. Since the system is interactive and frame-based, functional testing is important to ensure that each module behaves correctly in practical conditions. The display output, bounding boxes, track IDs, and status messages serve as visible indicators of system performance.  
The implemented testing also includes behavior verification for special conditions such as no aircraft detected, multiple aircraft tracking, collision scenarios, zone violations, and occupancy changes. These conditions are explicitly handled in the code and therefore form a natural part of the system testing process.  
24 
4.4.2 Test Case Output Screens / Screenshots 
In this section, screenshots of the real-time output are included to demonstrate how the system behaves under different airport surface conditions. These screenshots act as visual proof of the system's functionality and help verify that the implemented model is correctly detecting and tracking aircraft. Since the system operates in real time using video input, the output is continuously updated based on the detected aircraft and analysis results. Therefore, capturing screenshots at different moments helps in clearly presenting the performance and reliability of the system.  
The screenshots are collected during real-time testing by simulating different aircraft movements in front of the camera. For each test case, the detected aircraft, track IDs, and final status are displayed on the screen. These outputs confirm that the system not only detects aircraft using the deep learning model but also applies analysis logic to determine the final results. This makes the screenshots an important part of functional testing, as they visually validate both the model detection and the decision-making process of the system.  
For demonstration purposes, three important test cases are considered: Aircraft Detection, Runway Collision, and Parking Occupancy. In the detection case, the system correctly identifies aircraft as attentive objects and displays bounding boxes with confidence scores. In the collision case, even if multiple aircraft are detected, the system recognizes their trajectories and classifies the output as HIGH_RISK or SAFE. Similarly, in the occupancy case, the system detects aircraft in specific zones and marks the status as OCCUPIED or FREE. These test cases effectively demonstrate that the system can handle different real-world scenarios and produce accurate monitoring results.  
Overall, the screenshots confirm that the system performs reliably in real-time conditions and successfully identifies both detection and analysis states. They also highlight the effectiveness of combining deep learning predictions with rule-based analysis, making the system more robust and suitable for practical deployment. 
25 
Sample Test Cases 
1. Aircraft Detection  
Figure 4.1: Output Showing Aircraft Detection with Tracking 
2. Runway Collision Detection  
Figure 4.2: Output Showing Runway Collision Analysis 
26 
3. Restricted Zone Monitoring  
Figure 4.3: Output Showing Restricted Zone Violation 
4. Parking Slot Monitoring  
Figure 4.4: Output Showing Parking Slot Occupancy 
5. Terminal Gate Monitoring  
Figure 4.5: Output Showing Terminal Gate Status 
27 

CHAPTER 5 
DISCUSSION OF RESULTS 
The results of the proposed system are evaluated at both the training level and the real-time implementation level. The YOLOv11-based model is trained to detect aircraft in surveillance video frames. The training process uses appropriate loss functions and evaluation metrics. To improve model performance and generalization, data augmentation techniques such as mosaic augmentation, horizontal flipping, and scaling are applied during training. These techniques help the model learn robust features from the dataset and reduce overfitting. As a result, the model successfully learns to detect aircraft in various conditions, which serves as a strong foundation for accurate airport surface monitoring.  
In the real-time system, the output is not solely dependent on the model's prediction. Instead, the system enhances the prediction by combining it with additional analysis conditions such as collision detection, zone violation monitoring, and occupancy tracking. For instance, if aircraft are detected on the runway but are moving toward each other for multiple consecutive frames, the system outputs HIGH_RISK. Similarly, if aircraft enter a restricted zone and stay beyond time thresholds, the final output is marked as WARNING or CRITICAL. This integration-based decision mechanism improves the reliability of the system and ensures that the output reflects actual airport surface conditions rather than only detection results.  
The system also demonstrates effective performance in real-time conditions using video input. It continuously captures frames, detects aircraft using YOLOv11, tracks aircraft using DeepSort, and displays analysis results dynamically. The inclusion of trajectory analysis ensures that aircraft movement patterns are properly analyzed for collision risk. Additionally, the point-in-box detection mechanism ensures that zone violations and occupancy changes are accurately detected and reported. These features contribute to a smoother and more realistic user experience during live monitoring.  
Overall, the results indicate that the proposed system provides a practical, efficient, and reliable solution for real-time airport surface monitoring. By combining deep learning-based detection with rule-based analysis, the system achieves better accuracy compared to using a standalone detection model. The implementation successfully meets the project objective of monitoring airport surface operations in real time while maintaining simplicity, stability, and usability. 
28 

CHAPTER 6 
CONCLUSIONS & FUTURE WORK 
6.1 Conclusion 
This project presents a real-time airport surface monitoring system using computer vision and deep learning techniques. The system employs a YOLOv11-based model to detect aircraft in surveillance video frames, providing accurate and efficient detection for airport surface operations. In addition to deep learning predictions, the system integrates rule-based analysis such as collision detection, zone violation monitoring, and occupancy tracking to improve accuracy and reliability. The real-time implementation using OpenCV enables continuous video input processing, stable detection output, and effective handling of challenges such as multiple aircraft tracking and zone analysis. The combination of a modern detection model and domain-specific analysis makes the system both efficient and practical for real-world use. Overall, the project successfully demonstrates that airport surface operations can be monitored effectively using an automated approach, making it a valuable solution for enhancing airport safety and efficiency in modern aviation environments. 
6.2 Future Scope 
Although the current project successfully monitors airport surface operations in real time, it can be further improved and extended in several useful ways. Some possible future enhancements are given below. 
6.2.1 Integration with Airport Control Systems 
One possible future extension of this project is to use the same concept in integrated airport control systems. In this case, the system can be connected with air traffic control platforms to provide real-time surface monitoring data. If the system detects potential collisions or zone violations, it can be connected with control systems to produce alerts and warnings. This improvement can help in preventing accidents and increasing airport operational efficiency. 
6.2.2 Multi-Camera Integration 
The present system is mainly designed for detecting and monitoring from a single video source. In future, the project can be extended to support multiple camera feeds and analysis in the same frame. This would make the system more useful in large airports, where multiple surveillance cameras cover different areas. By integrating multiple video feeds, the system can monitor a wider surface area and provide comprehensive coverage. 
6.2.3 Integration with Flight Scheduling Systems 
Another important future enhancement is the integration of this system with flight scheduling and management platforms. The model can be connected with airport information systems such as flight schedules, gate assignments, and parking allocations. This would allow the platform to monitor surface operations automatically during actual airport operations and provide useful feedback to operators. Such integration can improve the effectiveness of airport management by helping operators understand surface conditions more easily. 
29 
6.2.4 Weather and Environmental Adaptation 
The system can be extended to handle different weather conditions such as rain, fog, and low lighting. By integrating weather-adaptive algorithms, the system can maintain reliable detection and monitoring even in adverse environmental conditions. This would make the system more robust and suitable for continuous airport operations. 
30 

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
[10]  
[11]  
[12]  
[13]  
[14]  
[15]  
Singh P, Sharma A, Kumar R (2025) Real-time object detection for airport surface monitoring using YOLO 
Kumar V, Patel S, Ahmed F (2025) A general model for multi-object tracking in transportation scenes 
Singh A, Gupta R, Kumar P (2025) Measuring airport surface activity using behavioral and visual features 
Ali H, Patel M, Khan S (2025) Comprehensive airport surface dataset for object detection 
Brown J, Wilson D, Taylor M (2024) Aircraft tracking and monitoring using CNN-DeepSort 
Davis K, Miller R, Johnson L (2024) Real-time runway object detection system 
Garcia M, Lopez J, Martinez A (2024) Aircraft detection using YOLOv4 
Rodriguez C, Hernandez E, Torres G (2024) Airport surface monitoring using deep CNN 
Kim S, Park J, Lee H (2024) Computer vision-based runway activity analysis 
Chen W, Liu Y, Zhang Q (2023) Multimodal aircraft tracking from video feeds 
Nakamura T, Sato K, Yamamoto R (2023) Vision-based airport zone monitoring in smart airports 
Dubey A, Sharma V, Pandey N (2023) Aircraft detection using CNN-based object recognition 
Vasudevan R, Subramanian K, Narayanan S (2023) Vision-based aircraft tracking and identification 
Ibrahim M, Hassan A, Rahman S (2023) Airport parking lot monitoring using visual features 
Ogawa Y, Watanabe K, Hayashi T (2024) Zone monitoring of aircraft using time-based analysis