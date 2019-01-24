function marginal_loglike(W::Matrix, d::Vector, err::Vector, Σ_X::Matrix; Σ_X_inv = inv(Σ_X), mu=Float64[], norm=1e18/size(W,2), ntot=())

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

function optimize_parameters(M::AxisymmetricEquilibrium, orbits::Vector, W::Matrix, d::Vector, err::Vector;
                             mu=Float64[], norm=1e18/size(W,2), ntot=(),
                             dr = (0.01,0.5), dE = (1.0,20.0), dp=(0.01,1.0),
                             batch_size=10000, atol=1e-3, maxiter = 100, alg = NelderMead(), kwargs...)

    pool = CachingPool(workers())
    if batch_size == 0
        bs = round(Int, length(orbits)/(5*nprocs()))
    end
    Js = pmap(x->get_jacobian(M, x; kwargs...), pool, orbits, on_error=x->zeros(length(x)), batch_size=bs)

    n = length(orbits)
    ns = length.(orbits)
    ts = [range(0.0, 1.0, length=nn) for nn in ns]
    Jis = [make_jacobian_spline(jj,tt) for (jj, tt) in zip(Js,ts)]
    orbs = [OrbitSpline(o) for o in orbits]

    f = function lml(Σ, Σ_inv, orbs, Jis, atol, pool, batch_size, W, d, err, norm, mu, ntot, sigma)
        Σ_p_inv = S44(inv(Diagonal(sigma.^2)))
        Σ .= compute_covariance_matrix(orbs, Jis, Σ_p_inv, atol, pool; batch_size=batch_size)
        Σ .= Σ + (1 + 1e-8)*abs(eigvals(Σ)[1])I #to make posdef
        Σ_inv .= inv(Σ)
        op = optimize(x -> -marginal_loglike(W, d, err, (10^x)*Σ; Σ_X_inv = (10^(-x))*Σ_inv,
                                             norm=norm,mu=mu,ntot=ntot), -6,3, Brent())
        println(sigma, " ", Optim.minimum(op))
        return Optim.minimum(op)
    end

    Σ = zeros(n,n)
    Σ_inv = zeros(n,n)
    m = f(Σ,Σ_inv, orbs, Jis, atol, pool, batch_size, W, d, err, norm, mu,  ntot, [10.0,0.2,0.1,0.1])
    println(m)
    #op = optimize(x -> f(Σ, Σ_inv, orbs, Jis, atol, pool, batch_size, W, d, err, norm, mu, ntot,
    #                     [x[1]*(dE[2]-dE[1])+dE[1],
    #                      x[2]*(dp[2]-dp[1])+dp[1],
    #                      x[3]*(dr[2]-dr[1])+dr[1],
    #                      x[3]*(dr[2]-dr[1])+dr[1]]),
    #              [0.5, 0.3, 0.2], alg,
    #              Optim.Options(show_trace=true, iterations=maxiter,show_every=1))

    #x = Optim.minimizer(op)
    config = ConfigParameters()         # calls initialize_parameters_to_default of the C API
    set_kernel!(config, "kMaternISO1")  # calls set_kernel of the C API
    config.sc_type = SC_MAP
    config.n_iterations=maxiter
    config.force_jump=1
    config.n_init_samples = 20
    config.init_method = 2
    config.n_iter_relearn = 10
    config.noise = 1e-12
    lowerbound = [0.0, 0.0, 0.0]; upperbound = [1.0, 1.0, 1.0]
    x, op = bayes_optimization(x -> (f(Σ, Σ_inv, orbs, Jis, atol, pool, batch_size, W, d, err, norm, mu, ntot,
                         [x[1]*(dE[2]-dE[1])+dE[1],
                          x[2]*(dp[2]-dp[1])+dp[1],
                          x[3]*(dr[2]-dr[1])+dr[1],
                          x[3]*(dr[2]-dr[1])+dr[1]]) - m), lowerbound, upperbound, config)

    sigma = [x[1]*(dE[2]-dE[1])+dE[1],
             x[2]*(dp[2]-dp[1])+dp[1],
             x[3]*(dr[2]-dr[1])+dr[1],
             x[3]*(dr[2]-dr[1])+dr[1]]

    Σ_x_inv = S44(inv(Diagonal(sigma.^2)))
    Σ .= compute_covariance_matrix(orbs, Jis, Σ_x_inv, atol, pool; batch_size=batch_size)
    Σ_inv .= inv(Σ)

    op = optimize(x -> -marginal_loglike(W, d, err, (10^x)*Σ; Σ_X_inv = (10^(-x))*Σ_inv,
                                         norm=norm, mu=mu, ntot=ntot), -6,3, Brent())
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
