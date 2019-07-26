clear; 
clc;
X1=importdata('prostate_train.txt');
X2=importdata('prostate_test.txt');

x=[ones(size(X1.data(:,1))) X1.data(:,1:4)];%²»¿¼ÂÇ½»²æÏî
y=X1.data(:,5);
[b,bint,r,rint,stats]=regress(y,x);
rcoplot(r,rint);
title('É¢µãÍ¼');
RSS= r'*r/67;

x2=[ones(size(X2.data(:,1))) X2.data(:,1:4)];%²âÊÔ¼¯Ô¤²â
y_hat=x2*b;
y2=X2.data(:,5);
err1= y2 -y_hat;
RSS1= err1'*err1/30;
plot(y2, err1, 'r^');

x=[ones(size(X1.data(:,1))) X1.data(:,1:4) X1.data(:,1).*X1.data(:,2) X1.data(:,1).*X1.data(:,3) X1.data(:,1).*X1.data(:,4) X1.data(:,2).*X1.data(:,3) X1.data(:,2).*X1.data(:,4) X1.data(:,3).*X1.data(:,4)];%¿¼ÂÇ½»²æÏî
y=X1.data(:,5);
[b,bint,r,rint,stats]=regress(y,x);
rcoplot(r,rint);
title('É¢µãÍ¼');
RSS= r'*r/67;

x2=[ones(size(X2.data(:,1))) X2.data(:,1:4) X2.data(:,1).*X2.data(:,2) X2.data(:,1).*X2.data(:,3) X2.data(:,1).*X2.data(:,4) X2.data(:,2).*X2.data(:,3) X2.data(:,2).*X2.data(:,4) X2.data(:,3).*X2.data(:,4)];%¿¼ÂÇ½»²æÏî
y_hat=x2*b;
y2=X2.data(:,5);
err1= y2 -y_hat;
RSS1= err1'*err1;
plot(y2, err1, 'r^');
