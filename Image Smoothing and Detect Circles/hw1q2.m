I1 = imread('cat_noise.bmp');
I11 = im2double(I1);

%2D Gaussian filter
G1 = fspecial('gaussian', [5,5], 1);
R1 = conv2(G1, I11);
G2 = fspecial('gaussian', [11,11], 3);
R2 = conv2(G2, I11);


%Filter seperability
[U,D,V] = svd(G1);
[U1,D1,V1] = svd(G2);
%Gaussian y direction
col = U(:,1) * sqrt(D(1,1));
col1= U1(:,1) * sqrt(D1(1,1));
%Gaussian x direction
row = V(:,1)' * sqrt(D(1,1));
row1 = V1(:,1)' * sqrt(D1(1,1));
%convolving rows and column to get gaussian filter
F1 = conv2(row, col);
F2 = conv2(row1,col1);

%convolving with the image
C1 = conv2( conv2( I11, row), col);
imwrite(R2,'cat_noise_seperability_5.bmp');

%convolving with the image
C2 = conv2( conv2( I11, row1), col1);
imwrite(R2,'cat_noise_seperability_11.bmp');

%difference between 3 time 3 filters since if the filters are same then their convolution results are same
filters_diff1 = max(max(G1-F1));
disp(filters_diff1);

%difference between the images 1 sigma
image_diff1 = max(max(R1-C1));
disp(image_diff1);

%difference between 11 times 11 filters since if the filters are same then their convolution results are same
filters_diff2 = max(max(G2-F2));
disp(filters_diff2);

%difference between the images 3 sigma
image_diff2 = max(max(R2-C2));
disp(image_diff2);


