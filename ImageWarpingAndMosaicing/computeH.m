function H = computeH(t1, t2)

%Input: Set of points t1 and t2 manually selected using cpselect

%Output: 3*3 Homography matrix

%Loading the required input files needed to run the program

%cpselect('sbu1.jpg', 'sbu2.jpg');
%load('movingPoints.mat');
%load('fixedPoints.mat');

%Code is expecting npoints * 2 matrices rather so need to transpose them
movingPoints = t1';
fixedPoints = t2';

%Number of pair of points manually selected using cpselect for the images
nPoints = size(movingPoints,1);

%For the Linear system of equations, Ah = 0. 
A = [];

for i=1:nPoints

    fixed = fixedPoints(i,:);
    moving = movingPoints(i,:);

    x2 = fixed(1);
    y2 = fixed(2);
    x1 = moving(1);
    y1 = moving(2);

    ax = [-x1,-y1,-1,0,0,0,x2*x1,x2*y1,x2];
    ay = [0,0,0,-x1,-y1,-1,y2*x1,y2*y1,y2];

    A = [A;ax;ay];
end

%We will find SVD of matrix A and the smallest singular value will be the
%values for Homography matrix
[~,~,V] = svd(A, 'econ');
H = V(:,9);

%Reshaping Homography marix to get 3*3 matrix
H = reshape(H, [3,3])';

end


%written by: Leena Shekhar