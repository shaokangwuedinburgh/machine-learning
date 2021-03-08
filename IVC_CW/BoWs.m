function image_feats = BoWs(imdsTrain,dictionary)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
vocab_size = size(dictionary, 2);
fprintf('\nbags of words\n')

num = size(imdsTrain.Files,1);
container = [];
step_p = 10;
binSize = 10;
image_feats = zeros(num,vocab_size);
for i = 1:num
    img = single(rgb2gray(imread(imdsTrain.Files{i})));
    input_img = vl_imsmooth(img,0.5);
    [~, sift_features] = vl_dsift(input_img,'Step',step_p,'size', binSize,'fast');
    sift_features = single(sift_features);
    %sift_features = vl_phow(img,'fast','true');
    %dist 是每一个keypoint对应在codebook里面的距离，距离越小，则越接近bag of words
    dist = vl_alldist2(sift_features,dictionary);  
    
    %找到sift feature对应的所有bag of words中，值最小的特征的下标(在codebook里面)
    %使用bag of words表示sift features, sift feature每有一个特征点在bag of words，就+1
    %即index(i)表示,可以用BoW里面的第i个特征表示该图片的第index(i)个特征点
    [~,index]=min(dist);
    
    %sift的所有特征中，可以用BoW中第i个表示的，有hist_v(i)个
    hist_v =histc(index,[1:1:vocab_size]);
    image_feats(i,:) = do_normalize(hist_v);
    if mod(i,30) ==0
        fprintf('\n image %d \n',i);
    end
end
end

