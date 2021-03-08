tic
%add Gaussian pixel noise
load('Resnet_Train1Val2Test3.mat');
num = length(imdsTest.Files);
% rng(12345);

Occlusion = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45];
% 
% Predictions = imdsTest.Labels;
% ACCURACY = [0,0,0,0,0,0,0,0,0,0];
% [label, score] = classify(net, imdsTest);
% ACCURACY(1)= accuracy(label, imdsTest);

for j=2:length(Occlusion)
    GenerateOcclusionSeed = fix(Occlusion(j)/2)+1;
    %每一个epoch下的图片，共享一个mask
    for i=1:num
        
        %产生随机数
        index1 = fix(GenerateOcclusionSeed + (224-2*GenerateOcclusionSeed).*rand);
        index2 = fix(GenerateOcclusionSeed + (224-2*GenerateOcclusionSeed).*rand);
        
        I = double(imread(imdsTest.Files{i}));
        Red = I(:,:,1);
        Green = I(:,:,2);
        Blue = I(:,:,3);
        
        
        if (mod(Occlusion(j),2))
            Red(index1-fix(Occlusion(j)/2):index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2):index2+fix(Occlusion(j)/2))=0;
            Green(index1-fix(Occlusion(j)/2):index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2):index2+fix(Occlusion(j)/2))=0;
            Blue(index1-fix(Occlusion(j)/2):index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2):index2+fix(Occlusion(j)/2))=0;
        end
        
        if (~mod(Occlusion(j),2))
            Red(index1-fix(Occlusion(j)/2)+1:index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2)+1:index2+fix(Occlusion(j)/2))=0;
            Green(index1-fix(Occlusion(j)/2)+1:index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2)+1:index2+fix(Occlusion(j)/2))=0;
            Blue(index1-fix(Occlusion(j)/2)+1:index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2)+1:index2+fix(Occlusion(j)/2))=0;
        end
        
        I(:,:,1) = Red;
        I(:,:,2) = Green;
        I(:,:,3) = Blue;

        %分类--------------------------------
        [label, score] = classify(net, I);
        Predictions(i) = label;
    end
    ACCURACY(j) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
    
end

toc

%%
%add Gaussian pixel noise
load('SVM_BoW_Train3Val1Test2.mat');
 
run('vlfeat-0.9.21/toolbox/vl_setup')
step_p = 10;
binSize = 10;
num = length(imdsTest.Files);
rng(12345);
vocab_size = size(dictionary, 2);
 
Occlusion = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]; 

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




for j=2:length(Occlusion)
    
    GenerateOcclusionSeed = fix(Occlusion(j)/2)+1;
    
    for i=1:num
        index1 = fix(GenerateOcclusionSeed + (224-2*GenerateOcclusionSeed).*rand);
        index2 = fix(GenerateOcclusionSeed + (224-2*GenerateOcclusionSeed).*rand);
        
        I = double(imread(imdsTest.Files{i}));
        Red = I(:,:,1);
        Green = I(:,:,2);
        Blue = I(:,:,3);
        
        
        if (mod(Occlusion(j),2))
            Red(index1-fix(Occlusion(j)/2):index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2):index2+fix(Occlusion(j)/2))=0;
            Green(index1-fix(Occlusion(j)/2):index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2):index2+fix(Occlusion(j)/2))=0;
            Blue(index1-fix(Occlusion(j)/2):index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2):index2+fix(Occlusion(j)/2))=0;
        end
        
        if (~mod(Occlusion(j),2))
            Red(index1-fix(Occlusion(j)/2)+1:index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2)+1:index2+fix(Occlusion(j)/2))=0;
            Green(index1-fix(Occlusion(j)/2)+1:index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2)+1:index2+fix(Occlusion(j)/2))=0;
            Blue(index1-fix(Occlusion(j)/2)+1:index1+fix(Occlusion(j)/2),index2-fix(Occlusion(j)/2)+1:index2+fix(Occlusion(j)/2))=0;
        end
        
        I(:,:,1) = Red;
        I(:,:,2) = Green;
        I(:,:,3) = Blue;
        
        
        TestImageFeatures = SingleImage(I, dictionary);
        [label,score] = predict(SVMModel,TestImageFeatures);
        Predictions(i)=string(label);
        
    end
    ACCURACY(j) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
    fprintf('%f',ACCURACY(j))
    toc
end