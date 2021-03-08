
%split the data into three parts, using rng to ensure produce same dataset.
imds = imageDatastore('catdog', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
rng(12345);
imds = shuffle(imds);
[Dataset1, Dataset2, Dataset3] = splitEachLabel(imds,0.33,0.33);