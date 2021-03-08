%Bag of Words
% imds = imageDatastore('catdog', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames'); 
% rng(12345);
% imds = shuffle(imds);
% [Dataset1, Dataset2, Dataset3] = splitEachLabel(imds,0.33,0.33);
%%
Dataset1 = imageDatastore('Dataset1/', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
Dataset2 = imageDatastore('Dataset2/', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
Dataset3 = imageDatastore('Dataset3/', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

rng(12345);
Dataset1 = shuffle(Dataset1);
Dataset2 = shuffle(Dataset2);
Dataset3 = shuffle(Dataset3);
%%

%Training on dataset1, validation on dataset 2, testing on dataset 3
imdsTrain = Dataset3;
imdsValidation = Dataset1;
imdsTest = Dataset2;
%%
num_of_words = 800; 
dictionary = cookbook(imdsTrain, num_of_words);

%% use Bag of words to represent the train images
% TrainImageFeatures = BoWs(imdsTrain, dictionary);
TestImageFeatures = BoWs(imdsTest, dictionary);
%%
run('vlfeat-0.9.21/toolbox/vl_setup')
TestLabels = MySVM(TrainImageFeatures, imdsTrain.Labels, TestImageFeatures);
%%
%evaluating the results
PreStrTestLabels = string(TestLabels);
TrueStrTestLabels = string(imdsTest.Labels);
%%
result = strcmp(PreStrTestLabels, TrueStrTestLabels);
accuracy = sum(result)/length(imdsTest.Files);