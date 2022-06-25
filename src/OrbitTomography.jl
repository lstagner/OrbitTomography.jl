__precompile__()

module OrbitTomography

using LinearAlgebra
using Statistics
using SparseArrays
using Distributed
using Printf
import Base.Iterators: partition

using Reexport
@reexport using GuidingCenterOrbits
using SciMLBase
using DiffEqBase
using OrdinaryDiffEq
using DifferentialEquations
using DiffEqGPU
#using DistributedArrays
using EFIT
using Equilibrium
using HDF5
using JLD2, FileIO
using Clustering
using Images
using StatsBase
using FillArrays
using ProgressMeter
using NearestNeighbors
using SharedArrays
using SparseArrays
using StaticArrays
using NonNegLeastSquares
using HCubature
using Interpolations
using Distributed
using Optim
using Sobol
using ForwardDiff
using Roots
using Plots
using Plots.PlotMeasures
using LaTeXStrings
import IterTools: nth

const S3 = SVector{3}
const S4 = SVector{4}
const S33 = SMatrix{3,3}
const S44 = SMatrix{4,4}

const e0 = 1.60217733e-19 # Coulombs / Joules
const mu0 = 4*pi*1e-7 # N/A^2
const c0 = 2.99792458e8 # m/s

const mass_u = 1.6605402e-27 # kg
const e_amu = 5.48579909070e-4 # amu
const H1_amu = 1.007276466879 # amu
const H2_amu = 2.0141017778 # amu
const H3_amu = 3.01550071632 # amu
const He3_amu = 3.01602931914 # amu
const B5_amu = 10.81 # amu
const C6_amu = 12.011 # amu

include("polyharmonic.jl")
export PolyharmonicSpline

include("spectra.jl")
export InstrumentalResponse, kernel
export ExperimentalSpectra, TheoreticalSpectra

include("io.jl")
export FIDASIMSpectra
export AbstractDistribution, FIDASIMGuidingCenterFunction, FIDASIMGuidingCenterParticles, FIDASIMFullOrbitParticles
export read_fidasim_distribution, write_fidasim_distribution
export FIDASIMBeamGeometry, FIDASIMSpectraGeometry, FIDASIMNPAGeometry, write_fidasim_geometry
export FIDASIMPlasmaParameters, FIDAWeightFunction, make_synthetic_weight_matrix

include("fidasim_utils.jl")
export split_spectra, merge_spectra, apply_instrumental!, merge_spectra_geometry
export split_particles, fbm2mc
export impurity_density, ion_density, weight_matrix

include("orbits.jl")
export OrbitGrid, orbit_grid, segment_orbit_grid,combine_orbits, fbm2orbit, mc2orbit
export map_orbits, bin_orbits
export write_orbit_grid, read_orbit_grid
export orbit_index, orbit_matrix
export OrbitSpline

include("psgrids.jl")
export PSGrid, DET_GCPtoEPR, getGCEPRCoord, fill_PSGrid, fill_PSGrid_batch, reconstruct_GCEPRCoords, reconstruct_PSGrid
export write_PSGrid, read_PSGrid, write_GCEPRCoords, read_GCEPRCoords, ps_VectorToMatrix, ps_MatrixToVector

include("gpu_grids.jl")
export fill_PSGridGPU

include("covariance.jl")
export epr_cov
export RepeatedBlockDiagonal, ep_cov, eprz_cov,transform_eprz_cov, eprz_kernel
export get_covariance, get_correlation, get_correlation_matrix, get_covariance_matrix
export get_global_covariance, get_global_covariance_matrix

include("weights.jl")
export AbstractWeight, FIDAOrbitWeight

include("tomography.jl")
export OrbitSystem, lcurve_point, lcurve, marginal_loglike, optimize_alpha!, estimate_rtol, optimize_parameters, inv_chol, solve

include("transforms.jl")
export EPDensity, local_distribution, RZDensity, rz_profile, EPRZDensity, eprz_distribution, epr2ps, epr2ps_splined, ps2epr, ps2epr_splined, ps2epr_sampled, epr2ps_covariance_splined
export orbsort, class_splines, psorbs_2_matrix, psorbs_2_matrix_INV, psorbs_2_matrix_DIST

include("analytic.jl")
export lnΔ_ee, lnΔ_ei, lnΔ_ii, slowing_down_time, critical_energy, approx_critical_energy
export gaussian, slowing_down, approx_slowing_down, bimaxwellian, maxwellian

include("basis.jl")
export Basis, construct_basis, evaluate

include("visualisation.jl")
export plot_PS_distribution_drop_dims, plot_PS_distribution_EP, plot_OG_distribution_drop_dims, plot_OG_distribution_ALL_E
export Overplot_topological_contour, Overplot_topological_contour2, plot_Energy_Dist, plot_topological_boundary, map_orbits_no_scaling

end # module
