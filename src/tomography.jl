mutable struct OrbitSystem{T}
    # sum(((W*f .- d)./err).^2) + (f .- mu)'*inv(alpha*S)*(f .- mu)
    W::Matrix{T}
    d::Vector{T}
    err::Vector{T}
    S::Matrix{T}
    S_inv::Matrix{T}
    T::Matrix{T}
    mu::Vector{T}
    alpha::T
end

function OrbitSystem(W, d, err, S; S_inv = inv(S), T=diagm(0=>ones(size(W,2))), mu = zeros(size(W,2)), alpha=1.0)
    OrbitSystem(W,d,err,S,S_inv,T,mu,alpha)
end

function marginal_loglike(OS::OrbitSystem; norm=1e18/size(OS.W,2))

    d = OS.d
    Σ_d = Diagonal(OS.err.^2)
    Σ_d_inv = inv(Σ_d)

    K = norm*OS.W

    mu_X = OS.mu/norm

    # Scale covariance matrices
    Σ_X = OS.alpha*OS.S
    Σ_X_inv = inv(OS.alpha)*OS.S_inv

    Σ_inv = Σ_X_inv .+ K'*Σ_d_inv*K
    X = Σ_inv \ (K'*(Σ_d_inv*d) + Σ_X_inv*mu_X)

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

function optimize_alpha!(OS; log_bounds = (-6,6), kwargs...)
    f = x -> begin
        OS.alpha = 10.0^x
        return -marginal_loglike(OS; kwargs...)
    end

    op = optimize(f, log_bounds[1], log_bounds[2], Brent())

    OS.alpha = 10.0^Optim.minimizer(op)

    return Optim.minimum(op)
end

function estimate_rtol(S)
    s = filter(x -> x > 0, eigvals(S))
    sn = log10.(reverse(s)/s[end])
    n = length(sn)
    x = collect(range(0,1,length=n))
    f = PolyharmonicSpline(3,x,sn,s=0.1)
    df2(t) = ForwardDiff.derivative(tt->ForwardDiff.derivative(f,tt),t)
    roots = find_zeros(df2,0.2,0.8)
    return 10.0^(f(roots[1]))
end

function optimize_parameters(make_orbit_system::Function, lbounds, ubounds;
                             niter=10, nseed = 15, warmstart=false, checkpoint=true,
                             file="optim_progress.jld2",verbose=true, kwargs...)

    s = Sobol.SobolSeq(lbounds,ubounds)
    points = Vector{Float64}[]
    values = Float64[]

    if warmstart && isfile(file) && (filesize(file) != 0)
        s, points, values = load(file, "s", "points", "values")
    end

    if checkpoint
        touch(file)
    end

    while s.s.n < nseed
        p = Sobol.next!(s)
        OS = make_orbit_system(p)
        v = optimize_alpha!(OS; kwargs...)
        push!(points, p)
        push!(values, v)
        if checkpoint
            @save file s points values
        end
        if verbose
            println("Seed Number: $(s.s.n)")
            println("Seed Point:  ", repr(p))
            println("Seed Value:  ", v)
            println("Best Seed: ", repr(points[argmin(values)]))
            println("")
        end
    end

    for i=1:niter
        spl = PolyharmonicSpline(2, Array(hcat(points...)'), values)
        xstart = s.lb .+ s.ub.*rand(length(s.ub))
        op = optimize(x -> spl(x...), s.lb, s.ub, xstart, Fminbox(NelderMead()))
        p = Optim.minimizer(op)
        pv = Optim.minimum(op)
        if p in points
            p = Sobol.next!(s)
            pv = spl(p...)
        end
        if verbose
            println("Iteration: $i")
            println("Guess Point:  ",repr(p))
            println("Guess Value:  ", pv)
        end
        OS = make_orbit_system(p)
        v = optimize_alpha!(OS; kwargs...)
        if isfinite(v)
            push!(points, p)
            push!(values, v)
        end
        if verbose
            println("Actual Value: ", v)
            println("Best Point: ", repr(points[argmin(values)]))
            println("")
        end
        if checkpoint
            @save file s points values
        end
    end

    #OS = make_orbit_system(points[argmin(values)])
    #optimize_alpha!(OS)
    return points[argmin(values)]#, OS
end

function inv_chol(Σ; rtol=0.0)
    if rtol == 0.0
        Σ_inv = inv(Σ)
        Γ = cholesky(Hermitian(Σ_inv)).U
    else
        S = svd(Σ)
        smax = maximum(S.S)
        svals_inv = Diagonal([s >= smax*rtol ? 1/s : 0.0 for s in S.S])
        Σ_inv = (svals_inv*S.Vt)' * S.U'
        Γ = sqrt.(svals_inv)*S.Vt
    end
    return Σ_inv, Γ
end

function solve(OS::OrbitSystem; rtol = 0.0, norm=1e18/size(OS.W,2), nonneg=true)

    d = OS.d
    err = OS.err
    Σ_d = Diagonal(err.^2)
    Σ_d_inv = inv(Σ_d)

    K = norm*OS.W

    mu_X = OS.mu/norm

    # Scale covariance matrices
    Σ_X_inv, Γ = inv_chol(OS.S; rtol=rtol)
    Σ_X = OS.alpha*OS.S
    Σ_X_inv *= inv(OS.alpha)
    Γ *= sqrt(OS.alpha)

    if nonneg
        try
            X = vec(nonneg_lsq(vcat(K./err, Γ), vcat(d./err, mu_X), alg=:fnnls))
        catch err
            @warn "Non-negative Least Squares failed. Using MAP estimate"
            println(err)
            Σ_inv = Σ_X_inv .+ K'*Σ_d_inv*K
            X = Σ_inv \ (K'*(Σ_d_inv*d) + Σ_X_inv*mu_X)
        end
    else
        Σ_inv = Σ_X_inv .+ K'*Σ_d_inv*K
        X = Σ_inv \ (K'*(Σ_d_inv*d) + Σ_X_inv*mu_X)
    end

    return norm*X
end
