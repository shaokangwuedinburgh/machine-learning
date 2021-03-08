load('Resnet_Result/Resnet_Train1Val2Test3_Q5_1.mat')
Resnet_Q5_1_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_1.mat')
Resnet_Q5_1_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_1.mat')
Resnet_Q5_1_3 = ACCURACY;

Resnet_Q5_1 = 1/3*(Resnet_Q5_1_1+Resnet_Q5_1_2+Resnet_Q5_1_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_1.mat')
SVM_Q5_1_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_1.mat')
SVM_Q5_1_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_1.mat')
SVM_Q5_1_3 = ACCURACY;

SVM_Q5_1 = 1/3*(SVM_Q5_1_1+SVM_Q5_1_2+SVM_Q5_1_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_1',STD,SVM_Q5_1)
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('Gaussian pixel noise')
%%
load('Resnet_Result/Resnet_Train1Val2Test3_Q5_2.mat')
Resnet_Q5_2_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_2.mat')
Resnet_Q5_2_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_2.mat')
Resnet_Q5_2_3 = ACCURACY;

Resnet_Q5_2 = 1/3*(Resnet_Q5_2_1+Resnet_Q5_2_2+Resnet_Q5_2_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_2.mat')
SVM_Q5_2_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_2.mat')
SVM_Q5_2_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_2.mat')
SVM_Q5_2_3 = ACCURACY;

SVM_Q5_2 = 1/3*(SVM_Q5_2_1+SVM_Q5_2_2+SVM_Q5_2_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_2',STD,SVM_Q5_2')
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('Gaussian blurring')
%%
clear
clc
load('Resnet_Result/Resnet_Train1Val2Test3_Q5_3.mat')
Resnet_Q5_3_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_3.mat')
Resnet_Q5_3_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_3.mat')
Resnet_Q5_3_3 = ACCURACY;

Resnet_Q5_3 = 1/3*(Resnet_Q5_3_1+Resnet_Q5_3_2+Resnet_Q5_3_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_3.mat')
SVM_Q5_3_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_3.mat')
SVM_Q5_3_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_3.mat')
SVM_Q5_3_3 = ACCURACY;

SVM_Q5_3 = 1/3*(SVM_Q5_3_1+SVM_Q5_3_2+SVM_Q5_3_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_3',STD,SVM_Q5_3')
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('Image contrast increase')
%%
clear
clc
load('Resnet_Result/Resnet_Train1Val2Test3_Q5_4.mat')
Resnet_Q5_4_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_4.mat')
Resnet_Q5_4_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_4.mat')
Resnet_Q5_4_3 = ACCURACY;

Resnet_Q5_4 = 1/3*(Resnet_Q5_4_1+Resnet_Q5_4_2+Resnet_Q5_4_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_4.mat')
SVM_Q5_4_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_4.mat')
SVM_Q5_4_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_4.mat')
SVM_Q5_4_3 = ACCURACY;

SVM_Q5_4 = 1/3*(SVM_Q5_4_1+SVM_Q5_4_2+SVM_Q5_4_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_4',STD,SVM_Q5_4')
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('Image contrast decrease')
%%
clear
clc
load('Resnet_Result/Resnet_Train1Val2Test3_Q5_5.mat')
Resnet_Q5_5_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_5.mat')
Resnet_Q5_5_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_5.mat')
Resnet_Q5_5_3 = ACCURACY;

Resnet_Q5_5 = 1/3*(Resnet_Q5_5_1+Resnet_Q5_5_2+Resnet_Q5_5_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_5.mat')
SVM_Q5_5_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_5.mat')
SVM_Q5_5_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_5.mat')
SVM_Q5_5_3 = ACCURACY;

SVM_Q5_5 = 1/3*(SVM_Q5_5_1+SVM_Q5_5_2+SVM_Q5_5_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_5',STD,SVM_Q5_5')
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('Image brightness increase')
%%
clear
clc
load('Resnet_Result/Resnet_Train1Val2Test3_Q5_6.mat')
Resnet_Q5_6_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_6.mat')
Resnet_Q5_6_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_6.mat')
Resnet_Q5_6_3 = ACCURACY;

Resnet_Q5_6 = 1/3*(Resnet_Q5_6_1+Resnet_Q5_6_2+Resnet_Q5_6_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_6.mat')
SVM_Q5_6_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_6.mat')
SVM_Q5_6_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_6.mat')
SVM_Q5_6_3 = ACCURACY;

SVM_Q5_6 = 1/3*(SVM_Q5_6_1+SVM_Q5_6_2+SVM_Q5_6_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_6',STD,SVM_Q5_6')
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('Image brightness decrease')
%%
clear
clc
load('Resnet_Result/Resnet_Train1Val2Test3_Q5_7.mat')
Resnet_Q5_7_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_7.mat')
Resnet_Q5_7_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_7.mat')
Resnet_Q5_7_3 = ACCURACY;

Resnet_Q5_7 = 1/3*(Resnet_Q5_7_1+Resnet_Q5_7_2+Resnet_Q5_7_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_7.mat')
SVM_Q5_7_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_7.mat')
SVM_Q5_7_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_7.mat')
SVM_Q5_7_3 = ACCURACY;

SVM_Q5_7 = 1/3*(SVM_Q5_7_1+SVM_Q5_7_2+SVM_Q5_7_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_7',STD,SVM_Q5_7')
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('HSV Hue noise increase')
%%
clear
clc
load('Resnet_Result/Resnet_Train1Val2Test3_Q5_8.mat')
Resnet_Q5_8_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_8.mat')
Resnet_Q5_8_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_8.mat')
Resnet_Q5_8_3 = ACCURACY;

Resnet_Q5_8 = 1/3*(Resnet_Q5_8_1+Resnet_Q5_8_2+Resnet_Q5_8_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_8.mat')
SVM_Q5_8_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_8.mat')
SVM_Q5_8_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_8.mat')
SVM_Q5_8_3 = ACCURACY;

SVM_Q5_8 = 1/3*(SVM_Q5_8_1+SVM_Q5_8_2+SVM_Q5_8_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_8',STD,SVM_Q5_8')
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('HSV Saturation noise increase')
%%
clear
clc
load('Resnet_Result/Resnet_Train1Val2Test3_Q5_9.mat')
Resnet_Q5_9_1 = ACCURACY;

load('Resnet_Result/Resnet_Train2Val3Test1_Q5_9.mat')
Resnet_Q5_9_2 = ACCURACY;

load('Resnet_Result/Resnet_Train3Val1Test2_Q5_9.mat')
Resnet_Q5_9_3 = ACCURACY;

Resnet_Q5_9 = 1/3*(Resnet_Q5_9_1+Resnet_Q5_9_2+Resnet_Q5_9_3)*100;


load('SVM_Result/SVM_BoW_Train1Val2Test3_Q5_9.mat')
SVM_Q5_9_1 = ACCURACY;

load('SVM_Result/SVM_BoW_Train2Val3Test1_Q5_9.mat')
SVM_Q5_9_2 = ACCURACY;

load('SVM_Result/SVM_BoW_Train3Val1Test2_Q5_9.mat')
SVM_Q5_9_3 = ACCURACY;

SVM_Q5_9 = 1/3*(SVM_Q5_9_1+SVM_Q5_9_2+SVM_Q5_9_3)*100;

STD = 0:9;
plot(STD,Resnet_Q5_9',STD,SVM_Q5_9')
legend('Resnet18','SVM+BoW')
xlabel('Standard Deviation')
ylabel('Accuracy')
title('Occlusion of the image increase')