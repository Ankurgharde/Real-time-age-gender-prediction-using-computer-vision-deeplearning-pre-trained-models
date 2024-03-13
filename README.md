# Real-time-age-gender-prediction-using-computer-vision-deeplearning-pre-trained-models
# INTRODUCTION:
- Gender and age play a significant role in interpersonal interactions among people who live in communities. The use of smart gadgets has expanded as technology has progressed, and social media has begun to draw everyone's attention.
- Daily studies on gender and age prediction have grown in prominence, it increases the number of apps that use such techniques. In these applications, facial photographs are commonly employed since they contain useful information that may be used to extract human interaction.
- For gender detection and age prediction, Image processing, feature extraction, and classification steps are usually used. These steps may change based on the objective of the study and the characteristics to be used.
- The face images were processed using a variety of approaches, and calculations were performed based on the results of the investigations. For image processing, there are two basic and typical which we need to follow.
- Image enhancement is the process of improving an image so that the resultant image is of higher quality and can be used by other applications. The most popular technique for extracting information from an image is the other technique.
- The image is divided into a specified number of parts or objects to solve the challenge and this procedure is called Segmentation.
- Due to the accuracy of its classification technique, deep learning techniques are a variety of tasks such as classification, feature extraction, object recognition, and so on, it helps in gender and age prediction.
The previous system’s machine learning algorithms were not utilized to improve classification skills for a vast number of images and data available via the internet.
- We might use age categories based on the experiment to forecast a person's age. The eyes are one of the most essential aspects of face images in various applications such as facial recognition and emotional expression.
- Human Facial Image Processing gives numerous hints and signs that can be applied to a variety of businesses, security, entertainment, and others. The expression on a person's face can reveal a lot of information about them, such as their emotional condition, the slightest agreement or disagreement, irony or fury, and so on.

# METHODOLOGY: 
# DEEP LEARNING
- Deep learning is an artificial intelligence (AI) technique that seeks to learn from experience to resemble the human brain. Through a training procedure, these 
  representations are learned.
- To teach the software how to detect an object, we must first train it with a large number of object images that we categorize according to different classes.
- Deep learning-based algorithms, on average, require a large amount of training data and take longer to train than traditional machine learning methods.
- Finding unique attributes when trying to recognize any object or character in an image is time-consuming and complex.
- Unlike traditional machine learning, where features are manually retrieved, problems can be solved using deep learning approaches, which extract important characteristics 
  from data automatically.
- A neural network with multiple hidden layers is known as deep learning. They may build complicated notions from simple concepts after an image has been taught over the 
  network.
- By integrating simple elements such as shape, edges, and corners, an image can be trained in the network to learn items such as characters, faces, and so on.
- As the image travels through the layers, each one gets a simple property while moving on to the next.
- As the layers grow larger, the network may learn more complex features and eventually merge them to identify the image.
- In the field of computer vision, deep learning has discovered a bunch of uses. The domains that work with facial data were among the most important computer vision 
  applications.
 
# CONVOLUTIONAL NEURAL NETWORK (CNN)
- A CNN (convolutional neural network) is a kind of artificial neural network that is commonly used for image or object identification and categorization.
- Using a CNN, Deep Learning recognizes items in an image. An input layer, hidden layers, and an output layer are all part of a standard neural network. The anatomy of the brain-inspired CNNs.
- Artificial neurons or nodes in CNNs collect inputs, process them, and deliver the result as output, rather like a neuron inside the brain functions and transmits signals between cells.
- The images are used as a source of data. Multiple hidden layers may exist in CNNs, each of which performs feature extraction from the image by performing calculations. The very first layer that extracts features out of an input image is convolution.
- The object is classified and identified in the output layer by the fully connected layer. The convolutional layer is the most important constituent of CNN. The mathematical procedure of convolution is used to combine two sources of data.
- Gender estimation from social image collection, images that do not require access to private details of the subject areas that are not displayed in the images, such as their birth date, and the usual approach that includes the collection of other information about an individual and on the basis about which we discover gender on manually handled annotated data for gender recognition. That is why we use D-CNN, which works directly on images and aids in precise gender estimation.
- Overfitting is usually a minor issue. This comes into play if deep learning or machine learning-based approaches are used on a dataset with such a small number of face images
 
# CNN MODEL
We must first extract the face from a webcam image before proceeding with the implementation. The OpenCV library in Python is used to accomplish this. An effective object detection method is face detection utilizing Haar feature-based cascade classifiers, which is a machine learning-based approach. There will be a large number of positive and negative photos, on which the classifier will be trained. It's then utilized to find faces in other pictures. Figure 2 shows a schematic illustration of the CNN model.

 
# Fig 2 CNN Diagram










# ALGORITHM FOR GENDER AND AGE PREDICTION 
We use Python Deep Learning in this study to detect the specific gender and age of provided facial data. Deep Learning is part of the machine learning category. Deep Learning is an Artificial Intelligence technology that mimics the functioning of human cognitive processing. From unstructured data collections, it can identify objects, people, talks, and characters. Input, Face Detection, Face Processing (Gender and Age classification), and Output are the four key sections of the algorithm
 
# Fig 3: Flowchart of Algorithm


# INPUT
- The main goal of this study is to make an entire system simpler and faster. There are a variety of ways to input the data into the algorithm to speed up the process. To begin, the user can utilize the system's webcam or another webcam digital device to collect data quickly.
# FACE DETECTION 
- A face recognition system is a bit of software that can match a human face in a video or digital image frame to a database of faces.
- Woody Bledsoe, Helen Chan Wolf, and Charles Bisson were among the first to develop facial recognition technology. Bledsoe, Wolf, and Bisson began working with computers to recognize human faces in 1964 and 1965.
- When detecting a face in a frame, some natural (lighting, posing angles, facial labeling) and digital (noise, interference) alterations are applied.
- Two properties of a human face as a template contribute to the difficulty of recognizing a human face: (1) The number of templates, or faces to be categorized, is enormous and almost certainly limitless. (2) Almost every pattern looks the same. We can use several sorts of audience records to fix this problem and keep the algorithm more efficient.
-  In neural networks, the audience set also acts as a standard for gender detection categorization.

#FACE PROCESSING: 

- After the face detection process, if a face is detected. A convolutional neural network, or CNN, can be used to begin processing.
- It is a kind of deep neural network that is primarily utilized for image processing. CNN goes through a training phase and makes a variety of estimations.
- It is a form of Deep Neural Network that is commonly used in image processing and natural language processing.
- The actual training phase will be carried out by CNN, and different predictions will be made.
- Male and female are the two genders that may be predicted. The challenge of estimating age is a multi-class task under which the periods are divided into groups.
- Because people of different ages have diverse face features, it's difficult to get precise data.
- We divided the population into age categories to speed up the procedure.
- The age estimation can fall into one of eight categories: (0–2), (4–6), (8–12), (15–20), (25–32), (38–43), (48–53), and (60–100).
# OUTPUT
The Login form will be provided as a start once we have launched the project using the Command Prompt. Once the credentials have been properly entered, the project window appears, which begins to identify if there is an object in front of the webcam, and if so, the algorithm classifies the gender type. Examples from our studies are shown below.
 
#Fig 4: Example of the output

# ACCURACY TESTING 
Once the technique has been deployed, we can begin evaluating it for accuracy. Commonly used procedure for testing accuracy is: 
- Input the data. 
- Create a frame.
- Detect the face.
- Process the image.
- Classify the Gender. 
- Classify the Age Group.
- Attach the result in the image.
- Output the image in the specified location

# LIMITATIONS 
Face identification has a severe challenge with skin color segmentation. The accuracy of facial segmentation is affected by the object's pose, noise, lighting conditions, and distance from the camera. The following are the numerous types of obstacles that may arise during detection:
- Pose
- Facial expression
- Imaging condition
- Age 
- Face size
- Different facial features 
- Illumination

# USE CASES
 - In a marketing organization, the target audience is identified.
 - During the recruitment process, to ensure the applicants' validity. 
 - Verification of the identity of those applying for government identification cards.
 - In the medical sector, a forensic department collects information about deceased people.
 - In the banking industry, age and gender detection can be used to extract information about an individual from photos.
 - The Criminal Investigation Department will compile information on the suspects based on their age and gender

# CONCLUSION
- Age and Gender Classification are two of the most essential resources for getting information from an individual. Human faces contain enough information to be useful for a 
  variety of purposes.
- Human age and gender classification are critical for reaching the right audience.
- We attempted to replicate the process using standard equipment.
- The algorithm's efficiency is determined by several factors, but the major goal of this study is to make it as simple and quick as possible while maintaining the highest 
  level of accuracy. Work is being done to improve the algorithm's efficiency.
- Future enhancements include discarding faces for non-human objects, adding more datasets for people of other ethnic groups, and giving the computer more granular control 
  over its workflow.
- Deep learning and CNN could be used to improve this prototype's ability to reliably identify a person's gender and age range out of a single image of their face.
- From this study, we can conclude with two important conclusions. First, despite the limited availability of age and gender-tagged photos, CNN can be used to improve age 
  and gender detection outcomes. Second, by employing additional training data and more complex systems, the system's performance can be slightly increased.


