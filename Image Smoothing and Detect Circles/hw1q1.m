%Gaussian filters with different sigma
G1 = fspecial('gaussian', [5,5], 1); 
G2 = fspecial('gaussian', [11,11], 3);

I1 = imread('cat_noise.bmp');
I11 = im2double(I1);
I2 = imread('cat.bmp');
I21 = im2double(I2);

%cat and cat noise 5 * 5
R1 = conv2(I11, G1);
imwrite(R1,'CatNoiseGaussian5.bmp');
R2 = conv2(I21, G1);
imwrite(R2,'CatGaussian5.bmp');

%cat and cat noise 11 * 11
R3 = conv2(I11, G2);
imwrite(R3,'CatNoiseGaussian11.bmp');
R4 = conv2(I21, G2);
imwrite(R4,'CatGaussian11.bmp');

I3 = imread('ducks_noise.bmp');
I31 = im2double(I3);
I4 = imread('ducks.bmp');
I41 = im2double(I4);

%ducks and duck noise 5 * 5
R5 = conv2(I31, G1);
imwrite(R5,'DucksNoiseGaussian5.bmp');
R6 = conv2(I41, G1);
imwrite(R6,'DucksGaussian5.bmp');

%ducks and ducks noise 11 * 11
R7 = conv2(I31, G2);
imwrite(R7,'DucksNoiseGaussian11.bmp');
R8 = conv2(I41, G2);
imwrite(R8,'DucksGaussian11.bmp');

I5 = imread('road_noise.bmp');
I51 = im2double(I5);
I6 = imread('road.bmp');
I61 = im2double(I5);

%road and road noise 5 * 5 
R9 = conv2(I51, G1);
imwrite(R9,'RoadNoiseGaussian5.bmp');
R10 = conv2(I61, G1);
imwrite(R10,'RoadGaussian5.bmp');

%road and road noise 11 * 11 
R11 = conv2(I51, G2);
imwrite(R11,'RoadNoiseGaussian11.bmp');
R12 = conv2(I61, G2);
imwrite(R12,'RoadGaussian11.bmp'); 



