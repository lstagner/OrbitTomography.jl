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

function optimize_parameters(M::AxisymmetricEquilibrium, orbits::Vector, W::Matrix, d::Vector, err::Vector;
                             mu=Float64[], norm=1e18/size(W,2), ntot=(),
                             batch_size=1000, atol=1e-3, maxiter = 200, kwargs...)

    pool = CachingPool(workers())
    Js = pmap(x->get_jacobian(M, x; kwargs...), pool, orbits, on_error=x->zeros(length(x)), batch_size=batch_size)

    ns = length.(orbits)
    ts = [range(0.0, 1.0, length=nn) for nn in ns]
    Jis = [make_jacobian_spline(jj,tt) for (jj, tt) in zip(Js,ts)]
    orbs = [OrbitSpline(o) for o in orbits]

    f = function lml(orbs, Jis, atol, pool, batch_size, W, d, err, norm, mu, ntot, sigma)
        Σ_inv = S44(inv(Diagonal(sigma.^2)))
        Σ = compute_covariance_matrix(orbs, Jis, Σ_inv, atol, pool; batch_size=batch_size)

        op = optimize(x -> -marginal_loglike(W, d, err, (10^x)*Σ; norm=norm,mu=mu,ntot=ntot),
                      -6,3, Brent())
        return Optim.minimum(op)
    end

    op = optimize(x -> f(orbs, Jis, atol, pool, batch_size, W, d, err, norm, mu, ntot, [x[1],x[2],x[3],x[3]]),
                  [10.0, 0.3, 0.1, 0.1], NelderMead(),
                  Optim.Options(show_trace=true, iterations=maxiter,show_every=10))

    sig = Optim.minimizer(op)
    sigma = [sig[1],sig[2],sig[3],sig[3]]
    Σ_x_inv = S44(inv(Diagonal(sigma.^2)))
    Σ = compute_covariance_matrix(orbs, Jis, Σ_x_inv, atol, pool; batch_size=batch_size)

    op = optimize(x -> -marginal_loglike(W, d, err, (10^x)*Σ; norm=norm, mu=mu, ntot=ntot),
                 -6,3, Brent())
    p = 10^Optim.minimizer(op)

    return p*Σ, vcat(p, sigma)
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
