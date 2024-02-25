% Define camera intrinsic matrix and radial distortion coefficients
IntrinsicMatrix = [1769.60561310104 0 0; 0 1763.89532833387 0; 1927.08704019384 1064.40054933721 1];
radialDistortion = [-0.244052127306437, 0.0597008096110524];

% Create a cameraParameters object with the defined parameters
cameraParams = cameraParameters('IntrinsicMatrix', IntrinsicMatrix, 'RadialDistortion', radialDistortion);

% List of camera directories to process
cameraDirs = {'C1', 'C2', 'C3', 'C4', 'C5'};

% Loop over each camera directory
for cDir = cameraDirs
    inputDir = fullfile('pose', cDir{1}); % Input directory for current camera
    outputDir = fullfile('pose', [cDir{1} '_undistort']); % Output directory for undistorted images
    
    % Ensure the output directory exists
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % Get a list of jpg files in the current directory
    files = dir(fullfile(inputDir, '*.jpg'));
    
    % Loop through each file in the directory
    for i = 1:numel(files)
        filename = files(i).name; % Get the current filename
        I = imread(fullfile(inputDir, filename)); % Read the current image
        
        % Undistort the image using the specified camera parameters
        J = undistortImage(I, cameraParams);
        
        % Write the undistorted image to the corresponding output directory
        imwrite(J, fullfile(outputDir, sprintf('%d.jpg', i)));
    end
end
