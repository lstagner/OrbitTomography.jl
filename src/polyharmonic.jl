struct PolyharmonicSpline
    dim::Int64
    order::Int64
    coeff::Vector{Float64}
    centers::Array{Float64,2}
    error::Float64
end

struct PolyharmonicSplineInv
    dim::Int64
    order::Int64
    coeff::Vector{Float64}
    centers::Array{Float64,2}
    error::Float64
end

struct PolyharmonicSplineNorm
    dim::Int64
    order::Int64
    coeff::Vector{Float64}
    centers::Array{Float64,2}
    error::Float64
    norms::Vector{Float64}
end

struct PolyharmonicSplineInvNorm
    dim::Int64
    order::Int64
    coeff::Vector{Float64}
    centers::Array{Float64,2}
    error::Float64
    norms::Vector{Float64}
end

function polyharmonicK(r,K)
    if iseven(K)
        iszero(r) && return zero(r)
        if r >= 1.0
            return (r^K)*log(r)
        elseif 0.0 < r < 1.0
            return (r.^(K-1))*log(r.^r)
        elseif iszero(r) # Needed for autodiff to work at zero
            return zero(r)
        end
    else
        return r^K
    end
end

function PolyharmonicSpline(K::Int64, centers::Array{Float64,2}, values::Array{Float64}; s = 0.0)
    m,n = size(centers)
    if n == length(values)
        return PolyharmonicSplineInv(K, centers, values; s = s)
    elseif m != length(values)
        throw(DimensionMismatch())
    end

    M = zeros(m,m)
    N = zeros(m,n+1)

    for i=1:m
        N[i,1] = 1
        N[i,2:end] = centers[i,:]
        for j=1:m
            M[i,j] = polyharmonicK(norm(centers[i,:] .- centers[j,:]),K)
        end
    end
    M = M + s*I
    L = vcat(hcat(M,N),hcat(N', zeros(n+1,n+1)))

    w = pinv(L)*vcat(values,zeros(n+1))

    ivalues = zeros(m)
    for i=1:m
        tmp = 0.0
        for j=1:m
            tmp = tmp + w[j]*polyharmonicK(norm(centers[i,:] .- centers[j,:]),K)
        end
        tmp = tmp + w[m+1]
        for j=2:n+1
            tmp = tmp + w[m+j]*centers[i,j-1]
        end
        ivalues[i] = tmp
    end
    error = norm(values .- ivalues)

    return PolyharmonicSpline(n,K,w,centers,error)
end

function PolyharmonicSplineInv(K::Int64, centers::Array{Float64,2}, values::Array{Float64}; s = 0.0)
    n,m = size(centers)

    if n == length(values)
        return PolyharmonicSpline(K, centers, values; s = s)
    elseif m != length(values)
        throw(DimensionMismatch())
    end

    M = zeros(m,m)
    N = zeros(m,n+1)

    for i=1:m
        N[i,1] = 1
        N[i,2:end] = centers[:,i]
        for j=1:m
            M[i,j] = polyharmonicK(norm(centers[:,i] .- centers[:,j]),K)
        end
    end
    M = M + s*I
    L = vcat(hcat(M,N),hcat(N', zeros(n+1,n+1)))

    w = pinv(L)*vcat(values,zeros(n+1))

    ivalues = zeros(m)
    for i=1:m
        tmp = 0.0
        for j=1:m
            tmp = tmp + w[j]*polyharmonicK(norm(centers[:,i] .- centers[:,j]),K)
        end
        tmp = tmp + w[m+1]
        for j=2:n+1
            tmp = tmp + w[m+j]*centers[j-1,i]
        end
        ivalues[i] = tmp
    end
    error = norm(values .- ivalues)

    return PolyharmonicSplineInv(n,K,w,centers,error)
end

function PolyharmonicSplineNorm(K::Int64, centers::Array{Float64,2}, values::Array{Float64}, norms::Vector{Float64}; s = 0.0)
    m,n = size(centers)
    if n == length(values)
        return PolyharmonicSplineInvNorm(K, centers, values, norms; s = s)
    elseif m != length(values)
        throw(DimensionMismatch())
    end

    length(norms)!=n && throw(DimensionMismatch())

    M = zeros(m,m)
    N = zeros(m,n+1)

    for i=1:m
        N[i,1] = 1
        N[i,2:end] = centers[i,:]
        for j=1:m
            M[i,j] = polyharmonicK(norm((centers[i,:] .- centers[j,:]) ./ norms),K)
        end
    end
    M = M + s*I
    L = vcat(hcat(M,N),hcat(N', zeros(n+1,n+1)))

    w = pinv(L)*vcat(values,zeros(n+1))

    ivalues = zeros(m)
    for i=1:m
        tmp = 0.0
        for j=1:m
            tmp = tmp + w[j]*polyharmonicK(norm((centers[i,:] .- centers[j,:]) ./ norms),K)
        end
        tmp = tmp + w[m+1]
        for j=2:n+1
            tmp = tmp + w[m+j]*centers[i,j-1]
        end
        ivalues[i] = tmp
    end
    error = norm(values .- ivalues)

    return PolyharmonicSplineNorm(n,K,w,centers,error,norms)
end

function PolyharmonicSplineInvNorm(K::Int64, centers::Array{Float64,2}, values::Array{Float64}, norms::Vector{Float64}; s = 0.0)
    n,m = size(centers)

    if n == length(values)
        return PolyharmonicSplineNorm(K, centers, values, norms; s = s)
    elseif m != length(values)
        throw(DimensionMismatch())
    end

    length(norms)!=n && throw(DimensionMismatch())

    M = zeros(m,m)
    N = zeros(m,n+1)

    for i=1:m
        N[i,1] = 1
        N[i,2:end] = centers[:,i]
        for j=1:m
            M[i,j] = polyharmonicK(norm((centers[:,i] .- centers[:,j]) ./ norms),K)
        end
    end
    M = M + s*I
    L = vcat(hcat(M,N),hcat(N', zeros(n+1,n+1)))

    w = pinv(L)*vcat(values,zeros(n+1))

    ivalues = zeros(m)
    for i=1:m
        tmp = 0.0
        for j=1:m
            tmp = tmp + w[j]*polyharmonicK(norm((centers[:,i] .- centers[:,j]) ./ norms),K)
        end
        tmp = tmp + w[m+1]
        for j=2:n+1
            tmp = tmp + w[m+j]*centers[j-1,i]
        end
        ivalues[i] = tmp
    end
    error = norm(values .- ivalues)

    return PolyharmonicSplineInvNorm(n,K,w,centers,error,norms)
end

function PolyharmonicSpline(K::Int64, centers::Vector{Float64},values::Vector{Float64};s = 0.0)
    PolyharmonicSpline(K,reshape(centers,length(centers),1),values,s=s)
end

function PolyharmonicSplineInv(K::Int64, centers::Vector{Float64},values::Vector{Float64};s = 0.0)
    PolyharmonicSpline(K,reshape(centers,1,length(centers)),values,s=s)
end

function (S::PolyharmonicSpline)(x::T...) where T <: Real
    n = length(x)
    n != S.dim && throw(DimensionMismatch("$n != $(S.dim)"))

    v = 0.0
    l = length(S.coeff)-(n+1)
    for j=1:l
        v = v + S.coeff[j]*polyharmonicK(norm(x .- S.centers[j,:]), S.order)
    end

    v = v + S.coeff[l+1]
    for j=2:n+1
        v = v + S.coeff[l+j]*x[j-1]
    end

    return v
end

function (S::PolyharmonicSplineInv)(x::T...) where T <: Real
    n = length(x)
    n != S.dim && throw(DimensionMismatch("$n != $(S.dim)"))

    v = 0.0
    l = length(S.coeff)-(n+1)
    for j=1:l
        v = v + S.coeff[j]*polyharmonicK(norm(x .- S.centers[:,j]), S.order)
    end

    v = v + S.coeff[l+1]
    for j=2:n+1
        v = v + S.coeff[l+j]*x[j-1]
    end

    return v
end

function (S::PolyharmonicSplineNorm)(x::T...) where T <: Real
    n = length(x)
    n != S.dim && throw(DimensionMismatch("$n != $(S.dim)"))

    v = 0.0
    l = length(S.coeff)-(n+1)
    for j=1:l
        v = v + S.coeff[j]*polyharmonicK(norm((x .- S.centers[j,:]) ./ S.norms), S.order)
    end

    v = v + S.coeff[l+1]
    for j=2:n+1
        v = v + S.coeff[l+j]*x[j-1]
    end

    return v
end

function (S::PolyharmonicSplineInvNorm)(x::T...) where T <: Real
    n = length(x)
    n != S.dim && throw(DimensionMismatch("$n != $(S.dim)"))

    v = 0.0
    l = length(S.coeff)-(n+1)
    for j=1:l
        v = v + S.coeff[j]*polyharmonicK(norm((x .- S.centers[:,j]) ./ S.norms), S.order)
    end

    v = v + S.coeff[l+1]
    for j=2:n+1
        v = v + S.coeff[l+j]*x[j-1]
    end

    return v
end