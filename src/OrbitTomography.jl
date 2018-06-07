__precompile__()

module OrbitTomography

using EFIT
using Equilibrium
using GuidingCenterOrbits
using HDF5
using Clustering
using Images
using StatsBase
using FillArrays

include("spectra.jl")
export InstrumentalResponse, kernel
export ExperimentalSpectra, TheoreticalSpectra

include("io.jl")
export FIDASIMSpectra, split_spectra, merge_spectra, apply_instrumental!
export AbstractDistribution, FIDASIMGuidingCenterFunction, FIDASIMGuidingCenterParticles, FIDASIMFullOrbitParticles
export read_fidasim_distribution, write_fidasim_distribution
export split_particles
export FIDASIMBeamGeometry, FIDASIMSpectraGeometry, FIDASIMNPAGeometry
export merge_spectra_geometry

include("covariance.jl")
export epr_cov
export RepeatedBlockDiagonal, ep_cov, eprz_cov

include("orbits.jl")
export orbit_grid, combine_orbits

include("weights.jl")
export AbstractWeight, FIDAOrbitWeight


end # module
