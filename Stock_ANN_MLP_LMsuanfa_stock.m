clc;
clear all;


whole_data = readmatrix('C:\xxxxxxxxxxxxxxxxxxx\SPY_20_years_data.csv');
%%%%%数据特点： 1 Date	2 Open	3 High	4 Low	5 Close	  6 Adj Close	7 Volume
raw_data = whole_data(:, [2, 3, 4, 5]);
%%%%%数据特点： 1 Open	2 High	3 Low	4 Close	

% numRows = height(raw_data);
% % 显示行数
% disp(['Table contains ', num2str(numRows), ' rows.']);
% %% 5033 rows

%大概看看数据长啥样
dataNormalized = zeros(size(raw_data));
for i = 1:4
    col = raw_data(:,i); % 获取第i列
    dataNormalized(:,i) = (col - min(col)) ./ (max(col) - min(col)); % 列内归一化
end
% 绘制归一化数据
figure; % 创建一个新的图形窗口
hold on; % 保持当前图形，以便在同一图中绘制多条线
colors = lines(4); % 生成14种不同的颜色
for i = 1:4
    plot(1:height(raw_data), dataNormalized(:,i), 'Color', colors(i,:), 'LineWidth', 0.3); % 绘制每列归一化数据
end
hold off;
legend('Open', 'Hight', 'Low', 'Close', 'Volume'); % 自定义图例文本
% 美化图表
title('Normalized raw data')
xlabel('Date days'); % 横坐标标签
ylabel('Value'); % 纵坐标标签
grid on; % 显示网格

%%%%%%%%%%%%%%%%%%
%用五分之四的数据进行训练， 用五分之一的数据进行预测
%输入参数可以是1 Open	2 High	3 Low	4 Close	的某个或某几个或全部
%输出参数是predictive close
inputs = dataNormalized(:, 1:3);
outputs = dataNormalized(:, 4);

% 创建一个MLP前馈网络，这里假设有5个神经元在隐藏层
% 你可以根据需要调整隐藏层神经元的数量
net = feedforwardnet(5, 'trainlm');%%train-lm is referred as lm or Levenberg–Marquardt algorithm

% 分割数据为训练集、验证集和测试集
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% LM里 可以调节的参数初始阻尼因子 λ ，最大迭代次数，迭代精度，收敛步长
net.trainParam.epochs = 99; % 设置最大迭代次数
% 开始时，可以设置一个相对较大的epochs值以确保算法有足够的时间收敛。
% 通过观察训练过程中的误差下降和验证集上的性能，
% 你可以调整这个值以防止过拟合或不必要的计算。如果发现模型很快就收敛了，
% 可以适当减少epochs；如果模型收敛太慢，可能需要增加epochs或调整其他参数。


% 过拟合后果泛化能力差：模型在新的数据上的预测性能下降，表现为验证集和测试集上的误差明显高于训练集上的误差。
net.trainParam.goal = 1e-3; % 设置训练目标精度
% goal参数设置为一个较小的正数，表示训练误差的目标值。如果设置得太低，可能导致过拟合，
% 因为模型会尝试过度拟合训练数据以达到非常低的误差率。开始时可以设置一个合理的目标精度，
% 基于实际问题的复杂度和所需的准确度。在训练过程中，根据实际训练误差和验证误差调整此值。
net.trainParam.mu_dec = 0.1; % LM算法中减小mu（阻尼因子）的比率
net.trainParam.mu_inc = 10; % LM算法中增加mu的比率
net.trainParam.mu_max = 1e10; % mu的最大值，阻尼因子的上限
net.trainParam.showWindow = true; % 训练过程中是否显示训练窗口

% 训练网络
[net,tr] = train(net,inputs',outputs');

% 使用训练好的网络进行预测
outputsPredict = net(inputs');

% 反归一化预测值以便于比较
maxClose = max(raw_data(:,4));
minClose = min(raw_data(:,4));
outputsPredict = outputsPredict' * (maxClose - minClose) + minClose;  %%预测的输出量
outputsReal = outputs' * (maxClose - minClose) + minClose; %%真实的输出量



% 绘制实际值与预测值
figure;
% plot(outputs * (maxClose - minClose) + minClose, 'b');
plot(outputsReal)
hold on;
plot(outputsPredict, 'r--');

legend('Actual Close','Predicted Close');
title('Stock Close Price Prediction');
xlabel('Days');
ylabel('Close Price');
grid on;










