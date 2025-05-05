clear all;
clc
fig_no = 1;

% Writing the data file for lammps initial condition

%%%%%% fixed paramters %%%%%%%%
filename = 'coords_init.dat';

natoms = 2000;                  % Number of cells
A0 = 1;                         % Preferred cell area 
A_avg = A0;                     % averaged cell area taken to be equal to preferred cell

Area_box = natoms*A_avg;        % actual area of simulation box
L = sqrt(Area_box);             % actual length      

% normalized quantities

l = L/sqrt(A0);          % Normalized box dimension 

x = l * rand(natoms, 1); % X-coordinates
y = l * rand(natoms, 1); % Y-coordinates

xcoord = x - l/2;        % center box
ycoord = y - l/2;
zcoord = zeros(natoms,1); 

xlo = -l/2;
xhi = l/2;
ylo = -l/2;
yhi = l/2;

r = [xcoord(:) ycoord(:)];           % update r
dt = delaunayTriangulation(r);       % use Delaunay initial triangulation
DT = delaunay(r);
[v,c] = voronoiDiagram(dt);          % voronoi for visualization

% Find the number of bonds in the system 

[VX,VY] = voronoi(r(:,1),r(:,2));

% Write datafile for lammps

atomID=(1:natoms)';
atom_type=ones(natoms,1);

fid1=fopen(filename,'w');
fprintf(fid1, '# Initialise atom positions\n');
fprintf(fid1,'\n');
fprintf(fid1, '%d atoms\n', natoms);
fprintf(fid1, '1 atom types\n');
fprintf(fid1,'\n');

fprintf(fid1, '%f %f xlo xhi\n', xlo, xhi);     
fprintf(fid1, '%f %f ylo yhi\n', ylo, yhi);      
fprintf(fid1, '%f %f zlo zhi\n', -0.5, 0.5);   %this is default for 2-D analysis in LAMMPS

fprintf(fid1,'\n');
fprintf(fid1, '%s\n','Masses');
fprintf(fid1,'%s\n','');
mass=1.0;                              %specify mass of atoms
fprintf(fid1,'\t\t1 %12.8f\n',mass);
fprintf(fid1,'Atoms\n');
fprintf(fid1,'\n');
fprintf(fid1, '%d %d %f %f %f\n', [double(atomID), atom_type, xcoord, ycoord, zcoord]'); %only rue for atom_type atomic
fclose(fid1);

%Plot
figure(fig_no)
voronoi(dt,'k',"LineWidth",1.5);
hold on;
scatter(r(:,1),r(:,2),5,'red','filled');

