I1 = imread('sac.bmp');
I11 = im2double(I1);

%horizontal
S1 = fspecial('sobel'); 

%vertical
S2 = S1';

%Sobel in horizontal direction
Ex = conv2(I11, S1);
%figure
%imshow(Ex); title('Sac Edge x');

%Sobel in vertical direction
Ey = conv2(I11, S2);
%figure
%imshow(Ey); title('Sac Edge y'); 

%Sobel combined
Exy = hypot(Ex,Ey);
%figure
%imshow(Exy); title('Sac Edge y'); 

%Max and Min 
disp(max(Exy(:)));
disp(min(Exy(:)));

R = imbinarize(Exy,.60); 
imwrite(R,'SacEdgeDetectionWithThresholding.bmp');





