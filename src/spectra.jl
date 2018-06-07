struct InstrumentalResponse{T<:Real}
    amp::Vector{T}
    width::Vector{T}    #in pixels
    location::Vector{T} #in pixels
end

function InstrumentalResponse()
    InstrumentalResponse(Float64[],Float64[],Float64[])
end

function (IR::InstrumentalResponse)(lambda,d=1.0)
    #d converts from pixels to wavelength
    sum([a*exp((-((d*l) - lambda)^2)/(d*w)^2) for (a,l,w) in zip(IR.amp,IR.location,IR.width)])
end

function kernel(lambda,IR::InstrumentalResponse,d=1.0)
    length(IR.location) == 0 && return Float64[]
    dL = lambda[2] - lambda[1]
    lmax = maximum(abs.(d*IR.location)) + 6*maximum(sqrt.(((d*IR.width).^2)/2))
    Lp = 0.0:dL:lmax
    L= vcat(-reverse(Lp)[1:end-1],Lp)
    k = IR.(L,d)
    return reflect(centered(k./sum(k))) #reflect for convolution not correlation
end

function kernel(lambda,IR::T,d=1.0) where T<:Real
    return reflect(Images.Kernel.gaussian((IR/d,), (21,)))
end

struct ExperimentalSpectra{T<:Real}
    lambda::Vector{T}
    data::Vector{T}
    err::Vector{T}
end

function Base.vcat(S::ExperimentalSpectra...)
    lambda = vcat((s.lambda for s in S)...)
    data = vcat((s.data for s in S)...)
    err = vcat((s.err for s in S)...)
    return ExperimentalSpectra(lambda,data,err)
end

struct TheoreticalSpectra{T<:Real}
    lambda::Vector{T}
    spec::Vector{T}
end

function Base.vcat(S::TheoreticalSpectra...)
    lambda = vcat((s.lambda for s in S)...)
    spec = vcat((s.spec for s in S)...)
    return TheoreticalSpectra(lambda,spec)
end

