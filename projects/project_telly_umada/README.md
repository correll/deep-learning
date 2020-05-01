# Final project 
Tetsumichi(Telly) Umada  
--- 

### project idea
Combine the output of a YOLO classifier with a word embedding to understand scene context

### model overview 
1. Use YOLO to detect objects (labels and location of objects) 
2. Craete feature vectors based on the outputs of the object 
3. Create a LSTM encoder 
4. Create a LSTM decoder to generate a caption


### data set  
For the experiment, the dataset is a flickr8k dataset. The dataset contains 6k training images, 1k dev images, and 1k test images. Each caption has 5 annotations for each image. 

### notebooks and slides
+ 0_YOLO.ipynb: Experiment/Explore the YOLO outputs. We use the pre-trained model for YOLO. 
+ 1_initial_model.ipynb: Initial model with some experiments. Some layers (Dense, Dropout, etc...). 
+ 2_yolo_lstm.ipynb: After creating feature vectors, they are put into the LSTM encoder.
+ presentation.pdf: model overivew and selected outputs 

### Reference 
Gentle guide on how YOLO Object Localization works with Keras (Part 2)              
https://heartbeat.fritz.ai/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-2-65fe59ac12d             

Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3               
https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088                     

How to Perform Object Detection With YOLOv3 in Keras             
https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/                 

Keras: The Python Deep Learning library             
https://keras.io/              

Getting started with the Keras functional API                
https://keras.io/getting-started/functional-api-guide/               

How to Develop a Deep Learning Photo Caption Generator from Scratch               
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/               

Multi-Modal Methods: Image Captioning (From Translation to Attention)                  
https://medium.com/mlreview/multi-modal-methods-image-captioning-from-translation-to-attention-895b6444256e                

