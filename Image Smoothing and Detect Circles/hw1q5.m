I1 = imread('sac.bmp');
I11 = im2double(I1);

%Mexican hat
G1 = fspecial('gaussian', [11,11], 1); 
L1 = [0, 1,0;1,-4,1;0,1,0];
C = conv2(G1, L1);

%Convolve with Mexican hat
R1 = conv2(C, I11);

%thresholding
[r,c] = size(R1);
threshold = .040;

for i=1:r-1
    for j=1:c-1
        if (((R1(i,j) >= 0 && R1(i,j+1)< 0) || (R1(i,j) <= 0 && R1(i,j+1)> 0)) && abs(R1(i,j)-R1(i,j+1)) > threshold)
            R1(i,j)=1;
        else
            R1(i,j)=0;
        end 
        if (((R1(i,j) >= 0 && R1(i+1,j)< 0) || (R1(i,j) <= 0 && R1(i+1,j)> 0)) && abs(R1(i,j)-R1(i+1,j)) > threshold)
            R1(i,j)=1;
        else
            R1(i,j)=0;
        end 
    end
end

H=hypot(R1,R2);
imwrite(H,'LoG.bmp');


