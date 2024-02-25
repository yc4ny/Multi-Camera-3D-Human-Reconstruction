% List of camera directories to process
cameraDirs = {'C1_undistort', 'C2_undistort', 'C3_undistort', 'C4_undistort', 'C5_undistort'};

% Loop over each camera directory
for cDir = cameraDirs
    % Generate the input and output directory paths
    inputDir = fullfile('checkerboard', cDir{1});
    outputDir = strrep(inputDir, 'undistort', 'keypoints');
    
    % Ensure the output directory exists
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    % Get a list of jpg files in the current input directory
    files = dir(fullfile(inputDir, '*.jpg'));
    % Sort files naturally to maintain numeric order
    files = natsortfiles({files.name});
    
    % Process each file in the directory
    for i = 1:numel(files)
        filename = files{i};
        % Read the current image
        I = imread(fullfile(inputDir, filename));
        
        % Detect checkerboard points in the image
        [imagePoints, boardSize] = detectCheckerboardPoints(I);
        
        % Display detected image points
        disp(imagePoints);
        
        % Save the detected points to a .mat file in the output directory
        save(fullfile(outputDir, sprintf('%d.mat', i)), 'imagePoints');
    end
end
