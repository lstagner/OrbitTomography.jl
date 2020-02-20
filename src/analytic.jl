function _gaussian(E::T, p::T, E0::T, p0::T; dE = 5.0, dthetap = 0.01) where T<:Number
    thetap = acos(p)/(2pi)
    thetap0 = acos(p0)/(2pi)
    p = clamp(p,nextfloat(-1.0),prevfloat(1.0))
    a = ((E - E0)/dE)^2 + ((thetap - thetap0)/dthetap)^2
    J = inv(sqrt(1.0 - p^2))/2pi
    return J*exp(-0.5*a)/(2pi*dE*dthetap)
end

function gaussian(energy::AbstractVector, pitch::AbstractVector, E0, p0; kwargs...)
    nenergy = length(energy)
    npitch = length(pitch)

    f = zeros(nenergy,npitch)
    for j=1:npitch, i=1:nenergy
        f[i,j] = _gaussian(energy[i],pitch[j], E0, p0; kwargs...)
    end
    return f
end

function lnΔ_ee(ne, Te)
    # Thermal electron-electron collisions
    # NRL Plasma Formulary ne in cm^-3, Te in eV
    return 23.5 - log(sqrt(ne)*(Te^(-5/4))) - sqrt(1e-5 + ((log(Te) - 2)^2)/16)
end

function lnΔ_ei(ne, Te, ni, Ti; mu = H2_amu, Z = 1)
    # Electron-ion collisions
    # NRL Plasma Formulary ne/i in cm^-3, Te/i in eV, mu in amu
    mr = e_amu/mu
    if Ti*mr < Te < 10Z^2
        return 23 - log(sqrt(ne)*Z*Te^(-3/2))
    elseif Ti*mr < 10Z^2 < Te
        return 24 - log(sqrt(ne)/Te)
    elseif Te < Ti*mr
        return 16 - log(mu*sqrt(ni)*(Ti^(-3/2))*Z^2)
    end
    return 16.0
end

function lnΔ_ii(ni1, Ti1, ni2, Ti2; mu1=H2_amu, mu2=H2_amu, Z1=1,Z2=1)
    # Mixed ion-ion collisions
    # NRL Plasma Formulary ni in cm^-3 Ti in eV, mu in amu

    return 23 - log(((Z1*Z2*(mu1+mu2))/(mu1*Ti2 + mu2*Ti1))*sqrt((ni1*Z1^2)/Ti1 + (ni2*Z2^2)/Ti2))
end

function lnΔ_ii(ne, Te, beta_D; mu1=H2_amu, mu2=H2_amu, Z1=1, Z2=1)
    # Counter-streaming ions (relative velocity v_D = beta_D*c) in the presence of warm electrons
    # NRL Plasma Formulary ne in cm^-3, Te in eV

    return 43 - log((Z1*Z2*(mu1 + mu2)/(mu1*mu2*beta_D^2))*sqrt(ne/Te))
end

function legendre(x::T, nterms) where T

    a = one(T)
    b = x

    p = zeros(T, nterms)

    for l=0:min(1,nterms-1)
        p[l+1] = l == 0 ? a : b
    end

    for l in 2:(nterms-1)
        a, p[l+1] = p[l], ((2l-1)*x*p[l] - (l-1)*a)/l
    end

    return p
end

function heaviside(t::T) where T
    t < 0 ? zero(T) : one(T)
end

function _slowing_down_legendre_expansion(u::T, P::Vector, P0::Vector) where T
    L = 0:(length(P)-1)
    S = 0.5*sum((2*l + 1) * P[l+1] * P0[l+1] * u^(l*(l+1)) for l in L)
    return S
end

function _slowing_down_legendre_expansion(xi::T, xi0::T, u::T, nterms) where T
    P = legendre(xi, nterms)
    P0 = legendre(xi0, nterms)
    _slowing_down_legendre_expansion(u, P, P0)
end

function _slowing_down(E::T, E0::T, P::Vector, P0::Vector; tau_off = 1.0, tau = 1.0,
                       th_amu = H2_amu, b_amu = th_amu, imp_amu = C6_amu,
                       Te = 1, Ti = Te, Zeff = 1, Zi=1, Zimp=6, Zb=Zi, ne = 1) where T<:Number
    # Te/Ti in keV; ne in cm^-3

    E >= E0  && return 0.0

    m_e = mass_u*e_amu
    m_b = mass_u*b_amu
    m_th = mass_u*th_amu
    m_imp = mass_u*imp_amu

    v = sqrt(2*(e0*E*1e3)/(m_b))
    v3 = v^3
    vb = sqrt(2*(e0*E0*1e3)/(m_b))
    vb3 = vb^3

    J = v*((e0*1e3)/m_b)

    nimp = Zeff > 1 ? ne*(Zeff - 1)/(Zimp*(Zimp-1)) : zero(ne)
    ni = max(ne - Zimp*nimp,zero(ne))

    lnΔ_e =     lnΔ_ei(ne, Te*1e3, ni, Ti*1e3; mu=b_amu, Z=Zi)
    lnΔ_i_i =   lnΔ_ii(ne, Te*1e3, vb/c0; mu1=th_amu, mu2=b_amu, Z1=Zi, Z2=Zb)
    lnΔ_i_imp = lnΔ_ii(ne, Te*1e3, vb/c0; mu1=imp_amu, mu2=b_amu, Z1=Zimp, Z2=Zb)

    # Calculate Slowing down time and critical energy/velocity
    # Heating of toroidal plasmas by neutral injection: T H Stix 1972 Plasma Phys. 14 367
    avg_cmr = (lnΔ_i_i*ni*(Zi^2/th_amu) + lnΔ_i_imp*nimp*(Zimp^2/imp_amu))/(ne*lnΔ_e)
    Ec = 14.8 * b_amu * Te * avg_cmr^(2/3)
    tau_s = 6.27e8 * b_amu*((Te*1e3)^1.5)/(ne*lnΔ_e*Zb^2)
    vc = sqrt(2*(e0*Ec*1e3)/m_b)
    vc3 = vc^3

    inv_v3vc3 = inv(v3 + vc3)
    u1 = v3/vb3
    u2 = (vb3 + vc3)*inv_v3vc3
    u = (u1*u2)^(Zeff/6)

    t_0 = tau_s*log(u2)/3
    t_b = tau_s*log((vb3+vc3)/vc3)/3
    t_1 = tau_off*t_b
    t = tau*t_b

    S = min(t,t_b,t_1)/t_b
    U = heaviside(t - t_0) - heaviside(t - t_1 - t_0)
    if U == 0
        return zero(T)
    end
    lex = _slowing_down_legendre_expansion(u, P, P0)
    g = max(J*S*(tau_s/(v3+vc3))*lex,zero(T))/t_b
end

function _slowing_down(E::T, p::T, E0::T, p0::T, nterms; kwargs...) where T<:Number
    P = legendre(p, nterms)
    P0 = legendre(p0, nterms)
    _slowing_down(E, E0, P, P0; kwargs...)
end

function slowing_down(energy::AbstractVector, pitch::AbstractVector, E0, p0; nterms=20, kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)

    f_slow = zeros(nenergy,npitch)
    P0 = legendre(p0, nterms)
    for j=1:npitch
        P = legendre(pitch[j], nterms)
        for i=1:nenergy
            f_slow[i,j] = _slowing_down(energy[i], E0, P, P0; kwargs...)
        end
    end

    return f_slow
end

function _approx_slowing_down(E::T, p::T, E0::T, p0::T;
                       th_amu = H2_amu, b_amu = th_amu, imp_amu = C6_amu,
                       Te = 1, Ti = Te, Zeff = 1, Zi=1, Zimp=6, Zb=Zi, ne = 1) where T<:Number
    # Te/Ti in keV; ne in cm^-3

    E >= E0  && return 0.0

    m_e = mass_u*e_amu
    m_b = mass_u*b_amu
    m_th = mass_u*th_amu
    m_imp = mass_u*imp_amu

    v = sqrt(2*(e0*E*1e3)/(m_b))
    v3 = v^3
    vb = sqrt(2*(e0*E0*1e3)/(m_b))
    vb3 = vb^3

    J = v*((e0*1e3)/m_b)

    nimp = Zeff > 1 ? ne*(Zeff - 1)/(Zimp*(Zimp-1)) : zero(ne)
    ni = max(ne - Zimp*nimp,zero(ne))

    lnΔ_e =     lnΔ_ei(ne, Te*1e3, ni, Ti*1e3; mu=b_amu, Z=Zi)
    lnΔ_i_i =   lnΔ_ii(ne, Te*1e3, vb/c0; mu1=th_amu, mu2=b_amu, Z1=Zi, Z2=Zb)
    lnΔ_i_imp = lnΔ_ii(ne, Te*1e3, vb/c0; mu1=imp_amu, mu2=b_amu, Z1=Zimp, Z2=Zb)

    # Calculate Slowing down time and critical energy/velocity
    # Heating of toroidal plasmas by neutral injection: T H Stix 1972 Plasma Phys. 14 367
    avg_cmr = (lnΔ_i_i*ni*(Zi^2/th_amu) + lnΔ_i_imp*nimp*(Zimp^2/imp_amu))/(ne*lnΔ_e)
    Ec = 14.8 * b_amu * Te * avg_cmr^(2/3)
    tau_s = 6.27e8 * b_amu*((Te*1e3)^1.5)/(ne*lnΔ_e*Zb^2)
    vc = sqrt(2*(e0*Ec*1e3)/m_b)
    vc3 = vc^3

    t_b = tau_s*log((vb3+vc3)/vc3)/3

    beta = (m_th*Zeff)/(2*m_b)
    alpha = (beta/3)*(1 - p0^2)*log((1 + vc3/v3)/(1+vc3/vb3))
    g = J*tau_s*inv(v3 + vc3)*inv(sqrt(4*pi*alpha))*exp(-((p - p0)^2)/(4*alpha))
    return g/t_b
end

function approx_slowing_down(energy::AbstractVector, pitch::AbstractVector, E0, p0; kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)

    f_slow = zeros(nenergy,npitch)
    for j=1:npitch, i=1:nenergy
        f_slow[i,j] = _approx_slowing_down(energy[i],pitch[j], E0, p0; kwargs...)
    end

    return f_slow
end

function _bimaxwellian(energy::S, pitch::S, T_perp::S, T_para::S;
                      vd_para=0.0, sigma=-1, th_amu = H2_amu) where S<:Number

    m_th = mass_u*th_amu

    T_perp = T_perp*1e3*e0
    T_para = T_para*1e3*e0
    Ed_para = (0.5*m_th*vd_para^2)

    E = energy*e0*1e3
    xi = clamp(sigma*pitch,nextfloat(-1.0),prevfloat(1.0))
    J = e0*1e3
    A = sqrt(E/(pi*T_para*T_perp^2))
    T1 = -(E*xi^2 + Ed_para - vd_para*xi*sqrt(2*m_th*E))/T_para
    T2 = -(1-xi^2)*E/T_perp
    return A*exp(T1 + T2)*J
end

function bimaxwellian(energy::AbstractVector, pitch::AbstractVector, T_perp, T_para; kwargs...)
    nenergy = length(energy)
    npitch = length(pitch)
    f_ep = zeros(nenergy, npitch)
    for j=1:npitch, i=1:nenergy
        f_ep[i,j] = _bimaxwellian(energy[i], pitch[j], T_perp, T_para; kwargs...)
    end
    T = (T_para + 2*T_perp)/3
    return f_ep
end

function maxwellian(energy, pitch, T; kwargs...)
    bimaxwellian(energy, pitch, T, T; kwargs...)
end
