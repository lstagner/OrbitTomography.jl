struct PolyharmonicSpline
    dim::Int64
    order::Int64
    coeff::Vector{Float64}
    centers::Array{Float64,2}
    error::Float64
end

function polyharmonicK(r,K)
    if iseven(K)
        return r < 1 ? (r.^(K-1))*log(r.^r) : (r^K)*log(r)
    else
        return r^K
    end
end

function PolyharmonicSpline(K::Int64, centers::Array{Float64,2}, values::Array{Float64}; s = 0.0)
    m,n = size(centers)
    m != length(values) && throw(DimensionMismatch())

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

function PolyharmonicSpline(K::Int64, centers::Vector{Float64},values::Vector{Float64};s = 0.0)
    PolyharmonicSpline(K,reshape(centers,length(centers),1),values,s=s)
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
