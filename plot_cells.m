
%% Code for visualizing cell monolayers and plotting histograms
clear all;
clc;

% Parameters 

Delta_t = 0.001;
nsteps = 50000;
nevery_out = 50;

set(0, 'DefaultAxesFontSize', 20);

%Load and parse data

data = readdump_all('cell_data.dump');

F = fieldnames(data); 
vec = data.(F{6});
xlimits = data.(F{3});
ylimits = data.(F{4});
natoms = size(vec,1);         % number of atoms
noutput = size(vec,3);        % number of times data outputted

ts = [0:nevery_out:nsteps];       % timesteps at which data is outputted 
time = ts*Delta_t;

vertices = read_file('vertices_list.txt');

% Set color map
numColors = 150; % Number of desired colors in the gradient
colorMap = parula(numColors);

k = 714;                  %choose time step

% extract data
xlo = xlimits(k,1);
xhi = xlimits(k,2);
ylo = ylimits(k,1);
yhi = ylimits(k,2);

r = [vec(:,2,k) vec(:,3,k)];   %position of atoms at kth time step
x_coord = r(:,1);
y_coord = r(:,2);
area = vec(:,6,k);
peri = vec(:,7,k);
shape_idx = peri./sqrt(area);
AR = vec(:,8,k);
maj_eigvec_x = vec(:,9,k);
maj_eigvec_y = vec(:,10,k);
energy = vec(:, 11,k);
sigma_xx = vec(:,12,k);
sigma_yy = vec(:,13,k);
sigma_xy = vec(:,14,k);
coordination = vec(:,15,k);
 
%% Plot cells colored with aspect ratio

    figure(4)
    colorMetric = 1-AR;   % set color metric
    hold on;
    colormap(parula(20)); % Choose colormap 
    clim([0 1]); 
    colorbar; 
    for i=1:natoms
        verts_i = vertices{k}{i}; 
        num_verts_coords = length(verts_i);
        vx = vertices{k}{i}(1:2:num_verts_coords);
        vy = vertices{k}{i}(2:2:num_verts_coords);
        patch(vx, vy, colorMetric(i), 'LineStyle', '-', 'LineWidth', 0.5, 'EdgeColor', 'k');
        hold on;
    end 
    axis equal;
    axis off;
    box on;
    axis([xlo xhi ylo yhi]);
    
%% Plot cells colored with orientation

    or = (maj_eigvec_x./sqrt(maj_eigvec_x.^2+maj_eigvec_y.^2)).^2;
    angle = atan2(maj_eigvec_y,abs(maj_eigvec_x));
    figure(4)
    colorMetric = angle;
    hold on;
    colormap(parula(20)); % Choose colormap 
    clim([0 1]); 
    colorbar; 
    for i=1:natoms
        verts_i = vertices{k}{i}; 
        num_verts_coords = length(verts_i);
        vx = vertices{k}{i}(1:2:num_verts_coords);
        vy = vertices{k}{i}(2:2:num_verts_coords);
        patch(vx, vy, colorMetric(i), 'LineStyle', '-', 'LineWidth', 0.5, 'EdgeColor', 'k');
        hold on;
    end 
    ax=gca;
    axis equal;
    axis off;
    box on;
    axis([xlo xhi ylo yhi]);
 
%% Plot histograms of desired cell properties
figure(1)
histogram(AR,14, 'FaceColor', [0, 0.0, 1.0])
xlabel('Aspect Ratio');

figure(2)
histogram(angle,14, 'FaceColor', [0, 0.0, 1.0])
xlabel('Coordination');

%% Calculate degree
for i = 1:natoms
    verts_i = vertices{k}{i};
    num_verts_coords = length(verts_i);
    vx = vertices{k}{i}(1:2:num_verts_coords);
    vy = vertices{k}{i}(2:2:num_verts_coords);
    deg(i) = length(vx);
end 

%% Plot 5-7 disclinations

 figure(2)
 for i=1:natoms
     verts_i = vertices{k}{i}; 
     num_verts_coords = length(verts_i);
     vx = vertices{k}{i}(1:2:num_verts_coords);
     vy = vertices{k}{i}(2:2:num_verts_coords);
     if num_verts_coords/2 == 5
        patch(vx,vy,'red','LineStyle','-','LineWidth',1.0);
     elseif num_verts_coords/2 == 7
        patch(vx,vy,'blue','LineStyle','-','LineWidth',1.0);
     else 
        patch(vx,vy,'white','LineStyle','-','LineWidth',1.0);  
     end
 hold on
 ax=gca;
 axis equal;
 axis off;
 box on;
 axis([xlo xhi ylo yhi]);
 end
    
%%
function data = read_file(filepath)
% Check if the file exists
if ~isfile(filepath)
    error('The file does not exist: %s', filepath);
end

% Open the file
fid = fopen(filepath, 'r');
if fid == -1
    error('Cannot open the file: %s', filepath);
end

% Initialize variables
data = {}; % Cell array to store data for each timestep
timestep = 0;
currentData = {}; % Temporary storage for the current timestep (cell array)

% Read the file line by line
while ~feof(fid)
    line = fgetl(fid); % Read a line
    if isempty(line) % Blank line separates timesteps
        if ~isempty(currentData)
            timestep = timestep + 1;
            data{timestep} = currentData; % Save the current timestep's data
            currentData = {}; % Reset for the next timestep
        end
    else
        % Parse the line as numeric data
        values = str2num(line); %#ok<ST2NM> % Dynamic length of columns
        if ~isempty(values)
            currentData{end+1, 1} = values; % Append row data as a cell array
        end
    end
end

% Save the last timestep's data if not already saved
if ~isempty(currentData)
    timestep = timestep + 1;
    data{timestep} = currentData;
end

% Close the file
fclose(fid);
end
