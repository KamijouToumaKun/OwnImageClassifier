# OwnImageClassifier
A multi-class image classifier that goes with your custom train set &amp; test set

## Path
You can modify the following default setting by giving different arguments to the model

    Your custom train set directory: "./train/1", "./train/2", ...
    Your custom test set directory: "./test"
    Output directory: "./test_out/1", "./test_out/2", ...
    Model directory: "./model"

## Environment
Python3 + TensorFlow.

## Performance
Best performance on my own dataset (not provided here:)
        
    Train Accuracy: 0.942604
    Test Accuracy: 0.8436658

You can modify the model that suits your custom dataset better.

## P.S.
You can add your custom label to the image file names if needed.

e.g. For Linux:

    cd directory
    for f in * ; do mv -- "$f" "yourlabel_$f" ; done
