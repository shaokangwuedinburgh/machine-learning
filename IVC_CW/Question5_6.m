tic
%add Gaussian pixel noise
load('Resnet_Train1Val2Test3.mat');
num = length(imdsTest.Files);
rng(12345);

BrightnessDecrease = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45];

Predictions = imdsTest.Labels;
ACCURACY = [0,0,0,0,0,0,0,0,0,0];
% for i=1:num
%     I = imread(imdsTest.Files{i});
%     I(:,:,1) = I(:,:,1) + noise;
%     I(:,:,2) = I(:,:,1) + noise;
%     I(:,:,3) = I(:,:,1) + noise;
% end
for j=1:length(BrightnessDecrease)
    
    for i=1:num
        
        I = double(imread(imdsTest.Files{i}));
        Red = I(:,:,1) - BrightnessDecrease(j);
        Green = I(:,:,2) - BrightnessDecrease(j) ;
        Blue =  I(:,:,3) - BrightnessDecrease(j) ;

%         Red(Red>255)=255;
        Red(Red<0)=0;

%         Green(Green>255)=255;
        Green(Green<0)=0;

%         Blue(Blue>255)=255;
        Blue(Blue<0)=0;

        I(:,:,1) = Red;
        I(:,:,2) = Green;
        I(:,:,3) = Blue;
        I = uint8(I);
        [label, score] = classify(net, I);
        Predictions(i) = label;
    end
    ACCURACY(j) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
    
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
 
BrightnessDecrease = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45];
Predictions = imdsTest.Labels;
ACCURACY = [0,0,0,0,0,0,0,0,0,0];
 
SVMModel = fitcsvm(TrainImageFeatures,imdsTrain.Labels);


for j=1:length(BrightnessDecrease)
    tic
%     flag = j;
%     ThisKernel = kernel;
%     tic
%     noise = double(normrnd(0,j,[224,224]));
    
    for i=1:num

         I = double(imread(imdsTest.Files{i}));
         Red = I(:,:,1) - BrightnessDecrease(j);
         Green = I(:,:,2) - BrightnessDecrease(j) ;
         Blue =  I(:,:,3) - BrightnessDecrease(j) ;

%         Red(Red>255)=255;
         Red(Red<0)=0;

%         Green(Green>255)=255;
         Green(Green<0)=0;

%         Blue(Blue>255)=255;
         Blue(Blue<0)=0;
        I(:,:,1) = Red;
        I(:,:,2) = Green;
        I(:,:,3) = Blue;
        
        TestImageFeatures = SingleImage(I, dictionary);
        [label,score] = predict(SVMModel,TestImageFeatures);
        Predictions(i)=string(label);
        
    end
    ACCURACY(j) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
    fprintf('\n %f \n',ACCURACY(j))
    toc
end
save(['SVM_Result/', 'SVM_BoW_Train1Val2Test3_Q5_6'])




clc
clear
load('SVM_BoW_Train2Val3Test1.mat');
 
run('vlfeat-0.9.21/toolbox/vl_setup')
step_p = 10;
binSize = 10;
num = length(imdsTest.Files);
rng(12345);
vocab_size = size(dictionary, 2);
 
BrightnessDecrease = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45];
Predictions = imdsTest.Labels;
ACCURACY = [0,0,0,0,0,0,0,0,0,0];
 
SVMModel = fitcsvm(TrainImageFeatures,imdsTrain.Labels);


% for ww=1:length(imdsTest.Files)
%     I = imread(imdsTest.Files{ww});
%     TestImageFeatures = SingleImage(I, dictionary);
%     [label,score] = predict(SVMModel,TestImageFeatures);
%     Predictions(ww)=string(label);
% end
% ACCURACY(1)=accuracy(Predictions, imdsTest);

for j=1:length(BrightnessDecrease)
    tic
%     flag = j;
%     ThisKernel = kernel;
%     tic
%     noise = double(normrnd(0,j,[224,224]));
    
    for i=1:num

         I = double(imread(imdsTest.Files{i}));
         Red = I(:,:,1) - BrightnessDecrease(j);
         Green = I(:,:,2) - BrightnessDecrease(j) ;
         Blue =  I(:,:,3) - BrightnessDecrease(j) ;

%         Red(Red>255)=255;
         Red(Red<0)=0;

%         Green(Green>255)=255;
         Green(Green<0)=0;

%         Blue(Blue>255)=255;
         Blue(Blue<0)=0;
        I(:,:,1) = Red;
        I(:,:,2) = Green;
        I(:,:,3) = Blue;
        
        
        TestImageFeatures = SingleImage(I, dictionary);
        [label,score] = predict(SVMModel,TestImageFeatures);
        Predictions(i)=string(label);
        
    end
    ACCURACY(j) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
    fprintf('\n %f \n',ACCURACY(j))
    toc
end
save(['SVM_Result/', 'SVM_BoW_Train2Val3Test1_Q5_6'])


clc
clear


load('SVM_BoW_Train3Val1Test2.mat');
 
run('vlfeat-0.9.21/toolbox/vl_setup')
step_p = 10;
binSize = 10;
num = length(imdsTest.Files);
rng(12345);
vocab_size = size(dictionary, 2);
 
BrightnessDecrease = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45];
Predictions = imdsTest.Labels;
ACCURACY = [0,0,0,0,0,0,0,0,0,0];
 
SVMModel = fitcsvm(TrainImageFeatures,imdsTrain.Labels);


% for ww=1:length(imdsTest.Files)
%     I = imread(imdsTest.Files{ww});
%     TestImageFeatures = SingleImage(I, dictionary);
%     [label,score] = predict(SVMModel,TestImageFeatures);
%     Predictions(ww)=string(label);
% end
% ACCURACY(1)=accuracy(Predictions, imdsTest);

for j=1:length(BrightnessDecrease)
    tic
%     flag = j;
%     ThisKernel = kernel;
%     tic
%     noise = double(normrnd(0,j,[224,224]));
    
    for i=1:num

        I = double(imread(imdsTest.Files{i}));
         Red = I(:,:,1) - BrightnessDecrease(j);
         Green = I(:,:,2) - BrightnessDecrease(j) ;
         Blue =  I(:,:,3) - BrightnessDecrease(j) ;

%         Red(Red>255)=255;
         Red(Red<0)=0;

%         Green(Green>255)=255;
         Green(Green<0)=0;

%         Blue(Blue>255)=255;
         Blue(Blue<0)=0;
        I(:,:,1) = Red;
        I(:,:,2) = Green;
        I(:,:,3) = Blue;
        
        
        TestImageFeatures = SingleImage(I, dictionary);
        [label,score] = predict(SVMModel,TestImageFeatures);
        Predictions(i)=string(label);
        
    end
    ACCURACY(j) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
    fprintf('\n %f \n',ACCURACY(j))
    toc
end
save(['SVM_Result/', 'SVM_BoW_Train3Val1Test2_Q5_6'])