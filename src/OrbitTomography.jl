__precompile__()

module OrbitTomography

using LinearAlgebra
using Statistics
using SparseArrays
using Distributed
using Printf
import Base.Iterators: partition

using EFIT
using Equilibrium
using GuidingCenterOrbits
using HDF5
using Clustering
using Images
using StatsBase
using FillArrays
using ProgressMeter
using NearestNeighbors
using SparseArrays
using StaticArrays
using NonNegLeastSquares
import IterTools: nth

include("spectra.jl")
export InstrumentalResponse, kernel
export ExperimentalSpectra, TheoreticalSpectra

include("io.jl")
export FIDASIMSpectra, split_spectra, merge_spectra, apply_instrumental!
export AbstractDistribution, FIDASIMGuidingCenterFunction, FIDASIMGuidingCenterParticles, FIDASIMFullOrbitParticles
export read_fidasim_distribution, write_fidasim_distribution
export split_particles, fbm2mc
export FIDASIMBeamGeometry, FIDASIMSpectraGeometry, FIDASIMNPAGeometry, write_fidasim_geometry
export merge_spectra_geometry

include("covariance.jl")
export epr_cov
export RepeatedBlockDiagonal, ep_cov, eprz_cov,transform_eprz_cov

include("orbits.jl")
export OrbitGrid, orbit_grid, segment_orbit_grid,combine_orbits, fbm2orbit, mc2orbit
export write_orbit_grid, read_orbit_grid, LocalDistribution, local_distribution
export orbit_index, orbit_matrix

include("weights.jl")
export AbstractWeight, FIDAOrbitWeight

include("tomography.jl")
export marginal_loglike, solve

end # module
