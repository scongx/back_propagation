function [res] = Predict_function(data,w1,w2,b1,b2)
%预测函数，以训练得到的参数w、b为基础，对新的归一化数据分析预测目标数据，最终返回预测值
%data为归一化后需要预测的数据，要求data行数为特征数，列数为样本数，w、b分别为训练得到的参数，hidd_num、output_num分别为隐训练模型隐藏层节点个数和输出层节点个数

%设定激活函数
syms x;
F = @(x) 1./(1+exp(-x));%激活函数设为sigmoid函数
f = @(x) exp(-x)./(exp(-x) + 1).^2;%f为F的导数

%设定模型隐藏层节点个数及输出层节点个数
hidden_num = 5;%可通过测试再确定隐藏层节点个数，调校后发现，隐藏层节点数接近输入层节点数时拟合效果较好
output_num = 1;


z1 = [];
z2 = [];
s1 = [];
s2 = [];

    for i = 1:size(data,2) %size(data,2)统计样本数量
        for h = 1:hidden_num
            z1(h,i) = data(:,i)'*w1(:,h)+b1(h);%z1为hidden_num*i的矩阵
            s1(h,i) = F(z1(h,i));%z1对应元素通过sigmoid函数激活后存储在s1矩阵中
        end
        
        z2(i) = s1(:,i)'*w2+b2;%z2为output_num*i的矩阵
        s2(i) = F(z2(i));%z2对应元素通过sigmoid函数激活后存储在s2矩阵中
        
    end
    
%res为需要返回的模拟目标数据赋值
res = s2;

end

