
clear;
clc;

%% Code for plotting simulation dynamics 

% Model Parameters
Delta_t = 0.001; 
nsteps1 = 0;
nsteps2 =50000;
nsteps = nsteps1+nsteps2;
nevery_out = 50;

% Load data
data = readdump_all('C:\Users\franck vernerey\Documents\Prakhar\DNiABM\cell_data.dump');

% Parse data
F = fieldnames(data);
vec = data.(F{6});
xlimits = data.(F{3});
ylimits = data.(F{4});
natoms = size(vec,1);         % number of atoms
noutput = size(vec,3);        % number of times data outputted

%
ts = [0:nevery_out:nsteps];       % timesteps at which data is outputted 
time = ts*Delta_t;


ts_log = [nsteps1:nevery_out:nsteps];
time_log = ts_log*Delta_t;

% Get ready to extract information
max_sigxx = 0;
min_sigxx = 0;
kk = nsteps1/nevery_out;

for k = 1:noutput                  %collect data from all ouput time steps

    r = [vec(:,2,k) vec(:,3,k)];   %position 
    area = vec(:,6,k);             
    peri = vec(:,7,k);
    shp_idx(k) = mean(peri./sqrt(area));
    energy = vec(:,8,k);
    sigma_xx = vec(:,12,k);
    sigma_yy = vec(:,13,k);
    coordination = vec(:,15,k);
    x_coord = r(:,1);
    y_coord = r(:,2);
    sig_xx_sum = 0.0;
    sig_yy_sum = 0.0;
    area_sum = 0.0;
    energy_tot(k) = 0.0;

    for i=1:natoms
        max_sigxx = max(max_sigxx, max(sigma_xx));        % Calculate totals
        min_sigxx = min(min_sigxx, min(sigma_xx));

        sig_xx_sum = sig_xx_sum + area(i)*sigma_xx(i);
        sig_yy_sum = sig_yy_sum + area(i)*sigma_yy(i);
        area_sum = area_sum + area(i);

        energy_tot(k) = energy_tot(k) + energy(i);
        
    end
    sig_xx_tot(k) = (1/area_sum)*sig_xx_sum;
    sig_yy_tot(k) = (1/area_sum)*sig_yy_sum;
    press(k) = (sig_xx_tot(k)+sig_yy_tot(k))/2;
    mean_coord(k) = mean(coordination);

    if (k > kk)
      sig_xx_tot_log(k-kk) = sig_xx_tot(k);
    end
end


%% Plot Simulation Dynamics 

blue = [0, 0.4470, 0.7410]; % Default Blue
red = [0.8500, 0.3250, 0.0980]; % Default Red

plot((time(100:end)-time(100)), shp_idx(100:end), 'LineWidth', 2.5); % Plot first line in blue
hold on
ylabel('Mean Shape Index'); % Label for left y-axis
h = plot((time(100:end)-time(100)),  5.099 * ones(size(time(100:end))), '--', 'Color', [blue, 0.7], 'LineWidth', 2.5); % 'k--' for black dashed line

% Set transparency (alpha)
h.Color(4) = 0.5; % 0 (fully transparent) to 1 (fully opaque)

legend(h, {' VGEF \langle q_i \rangle'}, 'Location', 'southeast');
lgd.FontSize = 10;
xlabel('Time'); % Label for x-axis
xlim([0,25])
grid on;




