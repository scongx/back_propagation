function [res,w1,w2,b1,b2] = bp_function(data,gar)
%训练函数，data为归一化后的训练数据，gar为归一化后的目标数据，u为学习速率，input、hidden、output分别为预设的输入层、隐藏层、输出层各层节点个数
%函数最终返回训练最终得到的w、b，res为模型最终模拟的目标数量

%确定输入层、隐藏层、输出层各层节点个数
input_num = 4;%4个特征值
hidden_num = 5;%可通过测试再确定隐藏层节点个数，调校后发现，隐藏层节点数接近输入层节点数时拟合效果较好
output_num = 1;

%权值阈值更新
u = 5.5;%调校后发现，当迭代次数为50次时，u为5到6时模型拟合结果较好

%随机初始化w、b
w1 = rand(input_num,hidden_num);%行数为输入层节点个数，列数为隐藏层节点个数
w2 = rand(hidden_num,output_num);%行数为隐藏层节点个数，列数为输出层节点个数
b1 = rand(hidden_num,1);%hidden_num*1的列向量
b2 = rand(output_num,1);%output_num*1的列向量

%设定激活函数
syms x;
F = @(x) 1./(1+exp(-x));%激活函数设为sigmoid函数
f = @(x) exp(-x)./(exp(-x) + 1).^2;%f为F的导数

%开始模型训练
z1 = [];
z2 = [];
s1 = [];
s2 = [];

for n = 1:1000 %迭代次数
    for i = 1:size(data,2) %size(data,2)统计样本数量
        for h = 1:hidden_num
            z1(h,i) = data(:,i)'*w1(:,h)+b1(h);%z1为hidden_num*i的矩阵
            s1(h,i) = F(z1(h,i));%z1对应元素通过sigmoid函数激活后存储在s1矩阵中
        end
        
        z2(i) = s1(:,i)'*w2+b2;%z2为output_num*i的矩阵
        s2(i) = F(z2(i));%z2对应元素通过sigmoid函数激活后存储在s2矩阵中
        
        
        %de为均方误差的导数，output_num*1的列向量
        de = gar(i) - s2(i);
        
        %db2是E对b2的修正，output_num*1的列向量
        db2 = f(z2(i))*de;
        
        %dw2是E对w2的修正，hidden_num*output_num的矩阵
        dw2 = s1(:,i)*db2';
        
        %fz1是激活函数对z1第i列元素求导得到的向量，fz1用于后续求E对w1的修正
        fz1 = f(z1(:,i));
        
        %让w2与db2对应元素相乘得到w2与db2的合并变量wf，用于之后dw1的计算
        
        wf = w2*db2;%hidden_num*output_num的矩阵
        
        %求E对b1的修正
        db1 = wf.*fz1;
        
        %求E对w1的修正
        for w = 1:hidden_num
            for ww = 1:input_num
                dw1(ww,w) = data(ww,i)*db1(w);
            end
        end
        
        %更新w、b
        w1 = w1 + u*dw1;
        w2 = w2 + u*dw2;
        b1 = b1 + u*db1;
        b2 = b2 + u*db2;
    end
end

%res为需要返回的模拟目标数据赋值
res = s2;

end
