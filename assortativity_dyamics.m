%% Modularity Dynamics Code %%

%% Read data

clear all;
clc;

% Load data

data = readdump_all('cell_data_long.dump');
neigh = readmatrix('neighs_list_long.txt','Delimiter',{'  '}); 

% Parameters
Delta_t = 0.001;
nsteps = 50000;
nevery_out = 50;
n = 2000;

ts = [0:nevery_out:nsteps];       % timesteps at which data is outputted 
time = ts*Delta_t;

% Extracting data
F = fieldnames(data); 
vec = data.(F{6});

%% Network analysis

% Choose timesteps 
first = 100;
last = 1001;

% Storage matrices 
r = zeros(last-first+1,1);
Q = zeros(last-first+1,1);
meanDiameter = zeros(last-first+1,1);
maxDiameter = zeros(last-first+1,1);
meanSize = zeros(last-first+1,1);
maxSize = zeros(last-first+1,1);

% Loop through snap shots and analyze networks
for k = 100:1001
    
    % Track progress
    if mod(k,50) == 0
        disp(k);
    end
    
    % Extract aspect ratio
    AR = vec(:,8,k);
    
    %Extract neighbor matrix
    neighborMatrix = neigh((k-1)*n+1:(k)*n,:);

    % Generate adjacency matrix
    A = zeros(n, n);
    for i = 1:size(neighborMatrix, 1)
        neighbors = neighborMatrix(i, :);
        neighbors = neighbors(~isnan(neighbors));  
        for j = 1:length(neighbors)
            A(i, neighbors(j)) = 1;
            A(neighbors(j), i) = 1; 
        end
    end
    
    % Set node attribute 
    x = AR;
    
    % Calculate assortativity coefficient
    m = sum(A(:)) / 2;
    kk = sum(A, 2); 
    
    K_outer = kk * kk';  
    X_outer = x * x';  
    
    num = sum(sum((A - K_outer/(2*m)) .* X_outer));
    
    delta = eye(size(A));
    
    den = sum(sum((diag(kk) - K_outer/(2*m)) .* X_outer));
    
    r(k) = num / den; % assortativity
    
    % Define cell classes and calculte modularity
    long = 1-x;
    long(long>0.65) = 2;
    long(long<=0.65) = 1;
    
    c=2;
    M = mix(A,long,c,m);   
    
    err = diag(M);
    ar = sum(M)';
    
    Qr = err - ar.^2;
    Q(k) = sum(Qr);            % modularity
    
    % Look at subgraph characteristics 
    id = 1 - x;
    sub = find(id>0.65);
    
    G = graph(A);
    
    % Create subgraph
    H = subgraph(G, sub);
    
    % Find connected components
    componentIndices = conncomp(H);
    numComponents = max(componentIndices);
    
    % Storage matrices
    diameters = zeros(1, numComponents);
    clusterSize = zeros(1,numComponents);
    
    for i = 1:numComponents
        nodesInComponent = find(componentIndices == i);
        clusterSize(i) = length(nodesInComponent);
        
        H_i = subgraph(H, nodesInComponent);
        
        D = distances(H_i);
        
        diameters(i) = max(D(~isinf(D)));
    end
    
    meanDiameter(k) = mean(diameters);      % remaining measures
    maxDiameter(k) = max(diameters);
    
    meanSize(k) = mean(clusterSize);
    maxSize(k) = max(clusterSize);
end


%% Generate plots of network measure dynamics

set(0, 'DefaultAxesFontSize', 40);

figure(1)

hold on
g = plot((time(100:end)-time(100)),  51 * ones(size(time(100:end))), '--', 'Color',[1, 0.0, 0.0,0.7], 'LineWidth', 3.0); 

% Set transparency (alpha)
g.Color(4) = 0.5; 

h = plot((time(100:end)-time(100)),  3 * ones(size(time(100:end))), '--', 'Color','k', 'LineWidth', 3.0); 

% Set transparency (alpha)
h.Color(4) = 0.5;
plot(time(100:end)-time(100),maxDiameter(100:end), 'LineWidth', 2.0, 'Color',  [0, 0.4470, 0.7410])
xlim([0,38])
title('LCC Diameter')

figure(2)

hold on
g = plot((time(100:end)-time(100)),  426 * ones(size(time(100:end))), '--', 'Color',[1, 0.0, 0.0,0.7], 'LineWidth', 3.0); 

% Set transparency (alpha)
g.Color(4) = 0.5;

h = plot((time(100:end)-time(100)),  5 * ones(size(time(100:end))), '--', 'Color','k', 'LineWidth', 3.0); 

% Set transparency (alpha)
h.Color(4) = 0.5; 
plot(time(100:end)-time(100),maxSize(100:end), 'LineWidth', 2.0, 'Color',  [0, 0.4470, 0.7410])
xlim([0,38]) 
title('LCC Size')

figure(3)


hold on
g = plot((time(100:end)-time(100)),  16.3387 * ones(size(time(100:end))), '--', 'Color',[1, 0.0, 0.0,0.7], 'LineWidth', 3.0); 
% Set transparency (alpha)
g.Color(4) = 0.5; 

h = plot((time(100:end)-time(100)),  1 * ones(size(time(100:end))), '--', 'Color','k', 'LineWidth', 3.0); 

% Set transparency (alpha)
h.Color(4) = 0.5; 
plot(time(100:end)-time(100), meanSize(100:end), 'LineWidth', 2.0, 'Color',  [0, 0.4470, 0.7410])

xlim([0,38]) 
title('Mean Cluster Size')

figure(4)


hold on
g = plot((time(100:end)-time(100)),  0.1185 * ones(size(time(100:end))), '--', 'Color',[1, 0.0, 0.0,0.7], 'LineWidth', 3.0);

% Set transparency (alpha)
g.Color(4) = 0.5; 

h = plot((time(100:end)-time(100)),  0.0079 * ones(size(time(100:end))), '--', 'Color','k', 'LineWidth', 3.0); 

% Set transparency (alpha)
h.Color(4) = 0.5; 
plot(time(100:end)-time(100), Q(100:end), 'LineWidth', 2.0, 'Color',  [0, 0.4470, 0.7410])
xlim([0,38]) 
title('Modularity')
ylim([0,.13])

%% Plot mixing matrix (by hand as needed)

imagesc(M);             % Display matrix as image
colormap(copper);       % Choose a colormap (e.g., parula, hot, jet)
colorbar;               % Show color scale
clim([0 1]);
axis off

%% Mixing matrix function

function [M] = mix(A, z, c, m)
    
    N = size(A, 1); 
    e_rs = zeros(c, c);
    
    for i = 1:N
        for j = i:N  
            r = z(i);
            s = z(j);
            e_rs(r, s) = e_rs(r, s) + A(i, j);
            e_rs(s, r) = e_rs(s, r) + A(i, j); 
        end
    end
    
    M = e_rs ./ (2*m);
end

