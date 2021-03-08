function test_labels = MySVM1(train_image, train_labels, test_image)
 
LAMBDA = 0.0001;
categories = unique(train_labels); 
num_categories = length(categories);
para_matrix = [];
for i = 1:num_categories
    
    categories1 = string(categories(i));
    train_labels1 = string(train_labels);
    matching_indices = strcmp(categories1, train_labels1);
    
%     matching_indices = strcmp(categories(i), train_labels);
    matching_indices = double(matching_indices);
    matching_indices(matching_indices~=1)= -1;
    [W B] = vl_svmtrain(train_image', matching_indices', LAMBDA);
    para_matrix= [para_matrix;[W',B]];  
    
end
 
numlabel = [];
for i = 1:num_categories
    label_emtpy = zeros(1,size(test_image,1));
    [~,~,~, scores] = vl_svmtrain(...
                        test_image', label_emtpy, 0, 'model',...
                        para_matrix(i,1:end-1)', 'bias', para_matrix(i,end), 'solver', 'none');
    
    numlabel = [numlabel,scores']; 
end
 
[~,index]=max(numlabel,[],2);
num = size(test_image,1);
test_labels = cell(num,1);
for i = 1:num
    test_labels{i}= categories(index(i)); 
end
% fprintf('\nsvm finished\n');
 
 
 
 
 


