## TensorFlow Implementation for HPC Project

This is a pure Tensorflow implemenation of the network described in the paper: A CATALOG OF VISUAL-LIKE MORPHOLOGIES IN THE 5 CANDELS FIELDS USING DEEP-LEARNING

Paper URL: http://arxiv.org/abs/1509.05429v1

A couple of notes:

1. This code is fairly ugly...
2. Ok now that were past that, usage: 
a. network.py is the main file, so to run:
   python network.py
b. You should have populated the directory data/jpeg_redundant_f160 with the f160 images
c. I need to separate the testing from the training so that the size of the datasets can be different.
3. Note, the f160 images have the dimensions 454x454x3, however in the paper it says that images need to be 45x45x3 so they are resized in datahelper.py 
4. Also, there isn't a standard max-out layer in tensorflow, which is what the paper reccomends for the fully connected layers, so I use standard ReLU. 

Finally, things are running, but there could be(and most likely are) bugs.





