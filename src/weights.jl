abstract type AbstractWeight end

struct FIDAOrbitWeight{T<:Real} <: AbstractWeight
    coordinate::EPRCoordinate
    lambda::Vector{T}
    weight::Vector{T}
end


