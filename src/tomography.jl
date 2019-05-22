function marginal_loglike(W::Matrix, d::Vector, err::Vector, Σ_X::Matrix; scale=1.0, rtol = 0.0,
                          Σ_X_inv = (rtol == 0.0 ? inv(Σ_X) : pinv(Σ_X, rtol=rtol)),
                          mu=Float64[], norm=1e18/size(W,2), ntot=())

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

    # Scale covariance matrices
    Σ_X *= scale
    Σ_X_inv *= inv(scale)

    Σ_inv = Σ_X_inv .+ K'*Σ_d_inv*K
    if length(mu) == 0
        X = Σ_inv \ (K'*(Σ_d_inv*d))
    else
        X = Σ_inv \ (K'*(Σ_d_inv*d) + Σ_X_inv*mu_X)
    end
    l = 0.0
    try
        dhat = K*X
        l = -logabsdet(Σ_X)[1] - logabsdet(Σ_inv)[1] -
            ((d .- dhat)'*Σ_d_inv*(d .- dhat))[1] - ((X .- mu_X)'*Σ_X_inv*(X .- mu_X))[1]
    catch er
        l = -Inf
    end

    return l
end

function inv_chol(Σ; rtol=0.0)
    if rtol == 0.0
        Σ_inv = inv(Σ)
        Γ = cholesky(Hermitian(Σ_inv)).U
    else
        S = svd(Σ)
        smax = maximum(S.S)
        svals_inv = Diagonal([s >= smax*rtol ? 1/s : 0.0 for s in S.S])
        Σ_inv = S.V*svals_inv*S.U'
        Γ = sqrt.(svals_inv)*S.V'
    end
    return Σ_inv, Γ
end

function solve(W::Matrix, d::Vector, err::Vector, Σ_X::Matrix; rtol = 0.0,
               mu=Float64[], norm=1e18/size(W,2), ntot=(), nonneg=true, scale=1.0)

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

    X = zeros(nc)
    Σ_X_inv, Γ = inv_chol(Σ_X; rtol=rtol)

    # Scale covariance matrices
    Σ_X *= scale
    Σ_X_inv *= inv(scale)
    Γ *= sqrt(inv(scale))

    if nonneg
        try
            X .= vec(nonneg_lsq(vcat(K./err, Γ), vcat(d./err, mu_X),alg=:fnnls))
        catch err
            @warn "Non-negative Least Squares failed. Using MAP estimate"
            println(err)
            Σ_inv = Σ_X_inv .+ K'*Σ_d_inv*K
            if length(mu) == 0
                X .= Σ_inv \ (K'*(Σ_d_inv*d))
            else
                X .= Σ_inv \ (K'*(Σ_d_inv*d) + Σ_X_inv*mu_X)
            end
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
