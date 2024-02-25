files = dir("checkerboard/C1_undistort/*.jpg");
files = natsortfiles(files);
for i = 1:numel(files)
    filename = files(i).name;
    I = imread("checkerboard/C1_undistort/"+filename);
    [imagePoints,boardSize] = detectCheckerboardPoints(I);
    disp(imagePoints)
    save( sprintf('checkerboard/C1_keypoints/%d.mat', i), 'imagePoints');

end

files = dir("checkerboard/C2_undistort/*.jpg");
files = natsortfiles(files);
for i = 1:numel(files)
    filename = files(i).name;
    I = imread("checkerboard/C2_undistort/"+filename);
    [imagePoints,boardSize] = detectCheckerboardPoints(I);
    disp(imagePoints)
    save( sprintf('checkerboard/C2_keypoints/%d.mat', i), 'imagePoints');

end

files = dir("checkerboard/C3_undistort/*.jpg");
files = natsortfiles(files);
for i = 1:numel(files)
    filename = files(i).name;
    I = imread("checkerboard/C3_undistort/"+filename);
    [imagePoints,boardSize] = detectCheckerboardPoints(I);
    disp(imagePoints)
    save( sprintf('checkerboard/C3_keypoints/%d.mat', i), 'imagePoints');

end

files = dir("checkerboard/C4_undistort/*.jpg");
files = natsortfiles(files);
for i = 1:numel(files)
    filename = files(i).name;
    I = imread("checkerboard/C4_undistort/"+filename);
    [imagePoints,boardSize] = detectCheckerboardPoints(I);
    disp(imagePoints)
    save( sprintf('checkerboard/C4_keypoints/%d.mat', i), 'imagePoints');

end

files = dir("checkerboard/C5_undistort/*.jpg");
files = natsortfiles(files);
for i = 1:numel(files)
    filename = files(i).name;
    I = imread("checkerboard/C5_undistort/"+filename);
    [imagePoints,boardSize] = detectCheckerboardPoints(I);
    disp(imagePoints)
    save( sprintf('checkerboard/C5_keypoints/%d.mat', i), 'imagePoints');

end

