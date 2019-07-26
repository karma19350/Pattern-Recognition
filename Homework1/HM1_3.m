function HM1_3
clc
clear
[X1,Y1]=generate(10,0.5);
%[X2,Y2]=generate(100,0.5);
[X3,Y3]=generate(100,0.5);

[X4,Y4]=generate(10,2);
%[X5,Y5]=generate(100,0.5);
[X6,Y6]=generate(100,2);
r=zeros(6,1);
R=zeros(6,1);
for j=1:3
    subplot(2,3,j);
    hold on;
    [~,r(j),R(j)]=simu(j,X1,Y1,X3,Y3);
    %[~,r(j),R(j)]=simu(j,X2,Y2,X3,Y3);
    hold off;
end
for j=1:3
    subplot(2,3,j+3);
    hold on;
    [~,r(j),R(j)]=simu(j,X4,Y4,X6,Y6);
    %[~,r(j),R(j)]=simu(j,X5,Y5,X3,Y3);
    hold off;
end
end


function [x, y]= generate(size,sigma)%输入样本数与标准差
theta1=3;
theta0=6;
x=zeros(size,1);
e=zeros(size,1);
for i=1:size
    x(i)=normrnd(0,1);
    e(i)=normrnd(0,sigma);
end
y=e+theta0+theta1*x;
end

function [ p,RSS1,RSS2 ] = simu(index,X,Y,X_n,Y_n)
p=polyfit(X,Y,index);%曲线拟合

y_hat=polyval(p,X);
err1= Y -y_hat;
RSS1= err1'*err1/100;

err=Y_n-polyval(p,X_n);
RSS2=err'*err/100;

%画图
x1 = linspace(min(min(X,min(X_n))),max(max(X,max(X_n))));
y1 = polyval(p, x1);
plot(X, Y, 'r^', x1, y1, 'k-',X_n,Y_n,'b.');
t=sprintf('RSS1=%.4f\nRSS2=%.4f',RSS1,RSS2);
text(max(max(X,max(X_n)))-1,polyval(p,min(min(X,max(X_n))))+1,t)
tit = char( vpa(poly2sym(p), 4));
title(tit);
end


