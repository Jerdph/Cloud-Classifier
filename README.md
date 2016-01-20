# ShapeGram
##Overview
ShapeGram is a classifier for 3D objects. The model extracts features from 3D printing files in .csv formats and classify the object into 6 classes, airplane, car, convertible car, helicopter, motorcycle, and train. I incorporated the model into a web-based game that allows player tocompete with the algorithm to classfiy 3D objects.

##Motivation
3D model classification has value in multiple industries and could be applied to areas such as architectural design, hardware QC, 
and autonomous driving. Another area I believe there is great value in is medical diagnosis where you can generate 3D files from MRI and CT scans, so you can potentially automate medical diagnosis to a certain degree. 
Since this project is my Galvanize Immersive Data Science Capstone, which has to be completed in 2 weeks, I decided to scope the project into building a 3D classifier for 6 classes with data that are readily available online.

##Data 
I am fortunate that I have access to data for all 6 classes(listed below) from [3Dwarehouse](https://3dwarehouse.sketchup.com/?hl=en).

1. Airplane
2. Car
3. Convertible Car
4. Helicopter
5. Motorcycle
6. Train

The data are originally in .skp format, which I converted into .csv so that i can be read into python pandas library. 

##Data Extraction
Since each 3D object are centered and scaled differently, I normalized the matrices and align them using 3D rotation matrices.
After the files are the same format, they were featurized using an algorithm built into the model. Featuring the model solve an issue that each files are different sizes and make the dimension of every files the same.

##Web App
To make things more fun, I created a web-based game that players can compete with my algorithm to classify 3D objects from point clouds. The web app has 3 pages which are described in detail below.
1. Landing Page

This is a landing page where you can view the game instruction and select difficulties. There are 3 levels of difficulty, easy, medium, and hard. These level dictates the number of points in the point cloud that will be plotted. The harder the difficulty the lower the number of points that will be plotted.

2. Game Page

3. Answer Page

##Instruction

