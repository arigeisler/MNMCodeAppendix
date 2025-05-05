# MNMCodeApppendix

A morphodynamic network model for cell monolayers: Appendix B: Code Repository

## Itemized list of contents:

#### Initial data generation code:

initial_data.m : Generates random distribution of points with Delaunay triangulation for an initial condition.

#### MNM custom LAMMPS model code: contains core model proposed in text, to be run with a full LAMMPS build, see LAMMPS documentation https://docs.lammps.org/

MNM_input_file.in : input file -- contains parameters for soft repulsion and viscosity regulation

core_MNM_model_header.h : header file

core_MNM_model.cpp : core c++ model code (implements model as proposed)

#### Data processing codes:

plot_sim_dynamics.m : processes data and plot simulation dyanmics (evolution of mean shape index)

plot_cells.m : processes data and plot cells with various colorings, plot histograms of cell properties

assortativity_dyamics.m : generates and plots network measures

