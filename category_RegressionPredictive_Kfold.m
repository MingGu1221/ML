% SVM分类预测 is developed for one physical quantity distribution at a single
% node independently

clc;
clear all;

whole_data = readtable('C:\Users\xxx\tableConvert.com_n0fwak.csv');
% 显示前几行数据以检查
% head(dataTable)
% 每列代表着 sepal_length	sepal_width	  petal_length	  petal_width	species

% 假设X是特征数据，每一行是一个样本，每一列是一个特征
% 假设Y是标签向量，其中包含每个样本的类别标签
% 例如，对于二分类问题，Y中的值可能为0和1

% 特征数据 & 标签
X = whole_data(:, 1: 4);
Y = categorical(whole_data.species);

% 训练集70%,
% 使用cvpartition函数和其HoldOut参数，数据集被分割成两个互斥的子集：一个用于训练模型的训练集和一个用于评估模型性能的测试集。
cv = cvpartition(Y, 'HoldOut', 0.3); % 保留30%的数据用于测试,分割数据集 没有validation集是因为后面用了k fold cross validation

idxTrain = training(cv);%要训练的索引
idxTest = test(cv);%要测试的

% 待训练
XTrainCV = X(idxTrain, :); %训练特征量
YTrainCV = Y(idxTrain, :); %表征到type里

XTest = X(idxTest, :);
YTest = Y(idxTest, :);

% 对于分类进行训练训练，计算k次迭代中指标(准确率)的平均，一般设置k=5or10
k = 10;
cv = cvpartition(YTrainCV, 'kfold', k);
% 初始化变量来存储每一折的准确率
accuracyList = zeros(cv.NumTestSets, 1);

for i = 1:cv.NumTestSets
    rng(1);  % 设置随机种子为1
    disp(rand(1));  % 生成一个随机数

    idxTrain = training(cv, i);%要训练的索引
    idxValidation = test(cv, i);%要Validation的索引

    XTrain = XTrainCV(idxTrain, :); %训练特征量
    YTrain = YTrainCV(idxTrain, :); 

    XValidation = XTrainCV(idxValidation, :);
    YValidation = YTrainCV(idxValidation, :); %验证

    % 多类分类： 将使用fitcecoc函数，它基于一对所有（One-vs-All）策略来训练逻辑回归模型。
    t = templateLinear('Learner', 'logistic');
    model = fitcecoc(XTrain, YTrain, 'Learners', t, 'Coding', 'onevsall', 'ObservationsIn', 'rows');
    
    % 使用训练好的模型对测试集进行预测。
    YPred = predict(model, XTest, 'ObservationsIn', 'rows');

% 计算准确率来使用 后续取平均
    accuracy = sum(YPred == YTest) / numel(YTest);
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
end

meanAccuracy = mean(accuracyList);
fprintf('Mean Accuracy after k-fold: %.2f%%\n', accuracy * 100);



