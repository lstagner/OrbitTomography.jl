function marginal_loglike(W::Matrix, d::Vector, err::Vector, Σ_X::Matrix; mu=Float64[], norm=1e18/size(W,2), ntot=())

    nr,nc = size(W)
    if length(mu) == 0
        mu_X = zeros(nc)
    else
        mu_X = mu/norm
    end

    if !isempty(ntot)
        W = vcat(W,ones(nc)')
        d = vcat(d,ntot[1])
        err = vcat(err, ntot[2]*ntot[1])
    end

    Σ_d = Diagonal(err.^2)
    Σ_d_inv = inv(Σ_d)

    K = norm*W

    Σ_X_inv = inv(Σ_X)
    Σ_inv = Σ_X_inv .+ K'*Σ_d_inv*K
    if length(mu) == 0
        X = Σ_inv \ (K'*(Σ_d_inv*d))
    else
        X = Σ_inv \ (K'*(Σ_d_inv*d) + Σ_X_inv*mu_X)
    end
    l = 0.0
    try
        dhat = K*X
        l = -nr*log(2pi) - logdet(Σ_d) - logabsdet(Σ_X)[1] - logabsdet(Σ_inv)[1] -
            ((d .- dhat)'*Σ_d_inv*(d .- dhat))[1] - ((X .- mu_X)'*Σ_X_inv*(X .- mu_X))[1]
    catch er
        l = -Inf
    end

    return l
end

function solve(W::Matrix, d::Vector, err::Vector, Σ_X::Matrix; mu=Float64[], norm=1e18/size(W,2), ntot=(), nonneg=true)

    nr,nc = size(W)
    if length(mu) == 0
        mu_X = zeros(nc)
    else
        mu_X = mu/norm
    end

    if !isempty(ntot)
        W = vcat(W,ones(nc)')
        d = vcat(d,ntot[1])
        err = vcat(err, ntot[2]*ntot[1])
    end

    Σ_d = Diagonal(err.^2)
    Σ_d_inv = inv(Σ_d)

    K = norm*W

    Σ_X_inv = inv(Σ_X)
    X = zeros(nc)
    if nonneg
        try
            Γ = cholesky(Hermitian(Σ_X_inv)).U
            X .= vec(nonneg_lsq(vcat(K./err, Γ), vcat(d./err, mu_X),alg=:fnnls))
        catch err
            @warn "Non-negative Least Squares failed. Using MAP estimate"
            Σ_inv = Σ_X_inv .+ K'*Σ_d_inv*K
            X .= Σ_inv \ (K'*(Σ_d_inv*d) + Σ_X_inv*mu_X)
        end
    else
        Σ_inv = Σ_X_inv .+ K'*Σ_d_inv*K
        if length(mu) == 0
            X .= Σ_inv \ (K'*(Σ_d_inv*d))
        else
            X .= Σ_inv \ (K'*(Σ_d_inv*d) + Σ_X_inv*mu_X)
        end
    end

    return norm*X
end
