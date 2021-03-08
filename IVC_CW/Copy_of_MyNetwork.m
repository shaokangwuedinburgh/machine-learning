%搭建自己的猫狗训练网络
net = resnet18;
imds = imageDatastore('catdog', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 

numTrainImages = numel(imds.Labels);

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

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

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
%%
options = trainingOptions('sgdm', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',1, ...
    'InitialLearnRate',0.1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,lgraph,options);

%%
[YPred,probs] = classify(net,augimdsValidation);
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