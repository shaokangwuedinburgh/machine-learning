function dictionary = cookbook(imdsTrain,num_of_words)
num = size(imdsTrain.Files,1);
container = [];
step_p = 10;
binSize = 10;
for i = 1:num
    img = single(rgb2gray(imread(imdsTrain.Files{i})));
    input_img = vl_imsmooth(img, 0.5);
    %sift_feature is a (128 * Number of key points) matrix
    %每一列是它的一个关键点的特征值
    [~, sift_features] = vl_dsift(input_img,'Step',step_p,'size', binSize,'fast');
    %sift_features = vl_phow(img,'fast','true');  
    %sift feature中每一个特征点有128维
    container =[container;(single(sift_features'))]; % 转置之后，每一行是一个关键点的特征值
    if mod(i,30) == 0
        fprintf('\n image %d \n',i);
    end
end


fprintf('\nstart to building vocaulary\n')
dictionary = vl_kmeans(container',num_of_words);
fprintf('\nfinish building vocaulary\n')
end

