tic
%add Gaussian pixel noise
load('Resnet_Train1Val2Test3.mat');
num = length(imdsTest.Files);
rng(12345);

kernel = 1/16*[1,2,1;2,4,2;1,2,1];

% Predictions = imdsTest.Labels;
% ACCURACY = [0,0,0,0,0,0,0,0,0,0];
% [label, score] = classify(net, imdsTest);
% ACCURACY(1) = accuracy(label, imdsTest);

% for i=1:num
%     I = imread(imdsTest.Files{i});
%     I(:,:,1) = I(:,:,1) + noise;
%     I(:,:,2) = I(:,:,1) + noise;
%     I(:,:,3) = I(:,:,1) + noise;
% end
for j=1:9
tic
    flag = j;
    ThisKernel = kernel;
    
%     while(flag~=1)
%         ThisKernel = conv2(ThisKernel, kernel);
%         flag = flag -1
%     end
        
    
    for i=1:num
        I = double(imread(imdsTest.Files{i}));

        Red = I(:,:,1);
        Green = I(:,:,2);
        Blue = I(:,:,3);
        
        
        while(flag~=0)
            Red = conv2(Red,ThisKernel, 'same');
            Green = conv2(Green,ThisKernel, 'same');
            Blue = conv2(Blue,ThisKernel, 'same');
            flag = flag - 1;
        end
        
        
        Red(Red>255)=255;
        Red(Red<0)=0;

        Green(Green>255)=255;
        Green(Green<0)=0;

        Blue(Blue>255)=255;
        Blue(Blue<0)=0;

        I(:,:,1) = Red;
        I(:,:,2) = Green;
        I(:,:,3) = Blue;
        I = uint8(I);
        [label, score] = classify(net, I);
        Predictions(i) = label;
    end
    ACCURACY(j+1) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
toc  
end

toc
%%
%add Gaussian pixel noise
load('SVM_BoW_Train1Val2Test3.mat');
 
run('vlfeat-0.9.21/toolbox/vl_setup')
step_p = 10;
binSize = 10;
num = length(imdsTest.Files);
rng(12345);
vocab_size = size(dictionary, 2);
 
 
noise = double(normrnd(0,0,[224,224]));
Predictions = imdsTest.Labels;
ACCURACY = [0,0,0,0,0,0,0,0,0,0];
 
SVMModel = fitcsvm(TrainImageFeatures,imdsTrain.Labels);


for ww=1:length(imdsTest.Files)
    I = imread(imdsTest.Files{ww});
    TestImageFeatures = SingleImage(I, dictionary);
    [label,score] = predict(SVMModel,TestImageFeatures);
    Predictions(ww)=string(label);
end
ACCURACY(1)=accuracy(Predictions, imdsTest);


kernel = 1/16*[1,2,1;2,4,2;1,2,1];


for j=1:9
    
    flag = j;
    ThisKernel = kernel;
    tic
    noise = double(normrnd(0,j,[224,224]));
    
    for i=1:num
        I = double(imread(imdsTest.Files{i}));

        Red = I(:,:,1);
        Green = I(:,:,2);
        Blue = I(:,:,3);
        
        
        while(flag~=0)
            Red = conv2(Red,ThisKernel, 'same');
            Green = conv2(Green,ThisKernel, 'same');
            Blue = conv2(Blue,ThisKernel, 'same');
            flag = flag - 1;
        end
        
        
        Red(Red>255)=255;
        Red(Red<0)=0;

        Green(Green>255)=255;
        Green(Green<0)=0;

        Blue(Blue>255)=255;
        Blue(Blue<0)=0;

        I(:,:,1) = Red;
        I(:,:,2) = Green;
        I(:,:,3) = Blue;
        
        
        TestImageFeatures = SingleImage(I, dictionary);
        [label,score] = predict(SVMModel,TestImageFeatures);
        Predictions(i)=string(label);
        
    end
    ACCURACY(j+1) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
    fprintf('%f',ACCURACY(j))
    toc
end