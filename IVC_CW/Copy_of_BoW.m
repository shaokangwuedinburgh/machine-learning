%Bag of Words
%分割数据集
imds = imageDatastore('catdog', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 


[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%%
num_of_words = 800; 
dictionary = cookbook(imdsTrain, num_of_words);

%% use Bag of words to represent the train images
TrainImageFeatures = BoWs(imdsTrain, dictionary);
TestImageFeatures = BoWs(imdsValidation, dictionary);
%%
run('vlfeat-0.9.21/toolbox/vl_setup')
TestLabels = MySVM(TrainImageFeatures, imdsTrain.Labels, TestImageFeatures);
%%
%evaluating the results
PreStrTestLabels = string(TestLabels);
TrueStrTestLabels = string(imdsValidation.Labels);
%%
result = strcmp(PreStrTestLabels, TrueStrTestLabels);
accuracy = sum(result)/length(imdsValidation.Files);