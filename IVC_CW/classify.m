%train a image
clc;
clear;
net = resnet18;
inputSize = net.Layers(1).InputSize;
classNames = net.Layers(end).ClassNames;
I = imread('catdog/DOGS/dog_12_99.png');
[label,scores] = classify(net,I);
figure
imshow(I)
title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");