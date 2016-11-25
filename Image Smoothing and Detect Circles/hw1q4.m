%Gaussian Filter
G1 = fspecial('gaussian', [11,11], 1); 

%Kernel 1
L1 = [0, 1,0;1,-4,1;0,1,0];
%Kernel 2
L2 = [1,1,1;1,-8,1;1,1,1];

%convolving with the 2 kernels
R1 = conv2(G1, L1);
R2 = conv2(G1, L2);

%Plot
[rows,columns]=size(R1);
x1=1:rows;
x2=1:columns;

figure();
[xg,yg]=meshgrid(x1,x2);
surf(xg,yg,R1);
figure();
surf(xg,yg,R2);




