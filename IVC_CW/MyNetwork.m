% %搭建自己的猫狗训练网络
% net = resnet18;
% 
% %split the data into three parts, using rng to ensure produce same dataset.
% imds = imageDatastore('catdog', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames'); 
% rng(12345);
% imds = shuffle(imds);
% [Dataset1, Dataset2, Dataset3] = splitEachLabel(imds,0.33,0.33);
% % imds = imageDatastore('catdog', ...
% %     'IncludeSubfolders',true, ...
% %     'LabelSource','foldernames'); 
% % 
% % numTrainImages = numel(imds.Labels);
% % 
% % [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
%%
net = resnet18;
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
inputSize = net.Layers(1).InputSize;

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);

%%
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

%%
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])
%%
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:end-3) = freezeWeights(layers(1:end-3));
lgraph = createLgraphUsingConnections(layers,connections);

%%
options = trainingOptions('sgdm', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',3, ...
    'InitialLearnRate',0.1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,lgraph,options);

%%
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end