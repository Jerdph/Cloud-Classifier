# ShapeGram
##Overview
ShapeGram is a classifier for 3D objects. The model extracts features from 3D printing files in .csv formats and classify the object into 6 classes, airplane, car, convertible car, helicopter, motorcycle, and train. I incorporated the model into a web-based game that allows player to compete with the algorithm to classify 3D objects.

##Motivation
3D model classification has value in multiple industries and could be applied to areas such as architectural design, hardware QC, 
and autonomous driving. Another area I believe there is great value in is medical diagnosis where, in addition to 3D imaging, you can generate 3D files from MRI and CT scans, so you can potentially automate medical diagnosis with machine learning to a certain degree. 
However since this project is my Galvanize Data Science Capstone, which has to be completed in 2 weeks, I decided to scope the project into building a 3D classifier for 6 classes with data that is readily available online.

##Data 
I am fortunate that I was able to find data for 6 classes (listed below) of transportation available from [3Dwarehouse](https://3dwarehouse.sketchup.com/?hl=en).

1. Airplane
2. Car
3. Convertible Car
4. Helicopter
5. Motorcycle
6. Train

The files are originally in .skp format, which I converted into .csv so that i can be read them using python pandas library. 

##Data Extraction

Since each 3D object are centered and scaled differently, I normalized the matrices and align them using 3D rotation matrices.
After the files are the same format, they were featurized using an algorithm built into the model into a feature matrix. Featurizing the model solved the issue that each files have different number of vertices by reducing their dimensions and making them the same.

##Modeling

I decided to use a support vector machine model with an RBF kernel. The model generated an accuracy of 81% with 5 folds cross validation and a 70% accuracy on unseen data set.

##Web App

To make things more fun, I created a web-based game that players can compete with my algorithm to classify 3D objects from point clouds. The web app has 3 pages which are described in detail below.

1. Landing Page
![alt tag](https://raw.github.com/jerdph/ShapeGram/master/img/home_page.png)

This is a landing page where you can view the game instruction and select difficulties. There are 3 levels of difficulty, easy, medium, and hard. These level dictates the number of points in the point cloud that will be plotted. The harder the difficulty the lower the number of points that will be plotted.

2. Game Page
![alt tag](https://raw.github.com/jerdph/ShapeGram/master/img/game_page.png)

The game will show you 3 different angular view of the point clouds of the same object. You can make your guess by clicking a button at the bottom of the screen

3. Answer Page
![alt tag](https://raw.github.com/jerdph/ShapeGram/master/img/result_page.png)

The result page tells you what object it was, whether you got it right, and whether you beat the algorithm. Happy playing!


##Instruction
If you did not download the data file provided here and wants to use your own set of data, you are welcome to do so, but please make sure the data folder is formatted the same way as the current structure because it is needed for the code to run properly.

1. To begin, run the create_plot.py file to generate all the plots necessary for the game. The script will read all the data files that are available in 'data/test set/' folder and will generate and save plots into 'code/static/temp/' folder where it will be automatically rendered by the program. You can add in as many data files as you like. 

2. Run create_model.py, this script will train the model using data in the data folder and pickle the model and scaler for data prediction.

3. Run flask_app.py, this will create the webapp at port 5000. Now you can visit the app and play the game!

Thanks for reading! Please feel free to give me any feedback at jerdph@gmail.com

Jerd
