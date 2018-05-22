abstract type AbstractWeight end

struct FIDAOrbitWeight <:AbstractWeight
    coordinate::EPRCoordinate
    weight::Vector{Float64}
end
