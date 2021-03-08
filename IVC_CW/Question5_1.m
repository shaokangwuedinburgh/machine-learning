tic
%add Gaussian pixel noise
load('Resnet_Train1Val2Test3.mat');
num = length(imdsTest.Files);
rng(12345);
noise = double(normrnd(0,0,[224,224]));
Predictions = imdsTest.Labels;
ACCURACY = [0,0,0,0,0,0,0,0,0,0];
for i=1:num
    I = imread(imdsTest.Files{i});
    I(:,:,1) = I(:,:,1) + noise;
    I(:,:,2) = I(:,:,1) + noise;
    I(:,:,3) = I(:,:,1) + noise;
end
for j=0:2:18
    noise = double(normrnd(0,j,[224,224]));
    for i=1:num
        I = double(imread(imdsTest.Files{i}));
        Red = I(:,:,1) + noise;
        Green = fix(I(:,:,2) + noise);
        Blue = fix(I(:,:,3) + noise);

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
    ACCURACY(j/2+1) = accuracy(Predictions, imdsTest);
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


noise = double(normrnd(0,0,[224,224]));
Predictions = imdsTest.Labels;
ACCURACY = [0,0,0,0,0,0,0,0,0,0];

SVMModel = fitcsvm(TrainImageFeatures,imdsTrain.Labels);

for j=0:2:18
    
    tic
    noise = double(normrnd(0,j,[224,224]));
    for i=1:num

            I = double(imread(imdsTest.Files{i}));
            Red = I(:,:,1) + noise;
            Green = fix(I(:,:,2) + noise);
            Blue = fix(I(:,:,3) + noise);

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
            I = single(rgb2gray(I));
            %         
            input_img = vl_imsmooth(I,0.5);
            [~, sift_features] = vl_dsift(input_img,'Step',step_p,'size', binSize,'fast');
            sift_features = single(sift_features);
            dist = vl_alldist2(sift_features,dictionary);  
            [~,index]=min(dist);
            %sift的所有特征中，可以用BoW中第i个表示的，有hist_v(i)个
            TestImageFeatures =histc(index,[1:1:vocab_size]);
            [label,score] = predict(SVMModel,TestImageFeatures);
%             TestLabels = MySVM1(TrainImageFeatures, imdsTrain.Labels, TestImageFeatures);
            Predictions(i)=string(label);
            %         
            end
    ACCURACY(j/2+1) = accuracy(Predictions, imdsTest);
    fprintf('\nfinish 1 epoch\n')
    toc
end
save(['SVM_Result/', 'SVM_BoW_Train3Val1Test2_Q5_1'])