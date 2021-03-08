function accuracy = accuracy(Prediction, imdsTest)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
Prediction = string(Prediction);
TrueLabel = string(imdsTest.Labels);
accuracy = sum(strcmp(Prediction, TrueLabel))/length(imdsTest.Labels);
end

