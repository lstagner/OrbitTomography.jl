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

function lnΔ_ii(ni1, Ti1, ni2, Ti2, ne, Te, beta_D; mu1=H2_amu, mu2=H2_amu, Z1=1, Z2=1)
    # Counter-streaming ions (relative velocity v_D = beta_D*c) in the presence of warm electrons
    # NRL Plasma Formulary ne in cm^-3, Te in eV

    m_i1 = mass_u*mu1
    m_i2 = mass_u*mu2
    m_e = mass_u*e_amu
    L1 = Ti1*e0/m_i1
    L2 = Ti2*e0/m_i2
    U = Te*e0/m_e

    if max(L1, L2) < (beta_D*c0)^2 < U
        return 43 - log((Z1*Z2*(mu1 + mu2)/(mu1*mu2*beta_D^2))*sqrt(ne/Te))
    else
        return lnΔ_ii(ni1, Ti1, ni2, Ti2; mu1=mu1, mu2=mu2, Z1=Z1, Z2=Z2)
    end
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

function slowing_down_time(ne, Te, Ti, Zeff;
                           Ai = H2_amu, Ab = H2_amu, Aimp=C6_amu,
                           Zi = 1, Zimp = 6, Zb=1)
    # Calculate Slowing down time on electrons
    # Heating of toroidal plasmas by neutral injection: T H Stix 1972 Plasma Phys. 14 367
    # Te in keV, ne in cm^-3

    nimp = Zeff > 1 ? ne*(Zeff - 1)/(Zimp*(Zimp-1)) : zero(ne)
    ni = max(ne - Zimp*nimp,zero(ne))

    lnΔ_e = lnΔ_ei(ne, Te*1e3, ni, Ti*1e3; mu=Ab, Z=Zb)

    tau_s = 6.27e8 * Ab*((Te*1e3)^1.5)/(ne*lnΔ_e*Zb^2)

    return tau_s
end

function _electron_ion_drag_difference(Eb, ne, Te, Ti, Zeff; Ab=H2_amu, Ai=H2_amu, Aimp=C6_amu, Zb=1,Zi=1,Zimp=6)
    # Calculates difference of electron drag - ion drag
    # Energetic ion distribution resulting from neutral beam injection in tokamaks: J Gaffey 1976 J. Plasma Phys. 16 149
    # E0/Te in keV, ne in cm^-3

    m_e = e_amu*mass_u
    m_b = Ab*mass_u
    m_imp = Aimp*mass_u
    m_i = Ai*mass_u

    v_b = sqrt(2*Eb*e0*1e3/m_b)
    v_e = sqrt(2*Te*e0*1e3/m_e)

    nimp = Zeff > 1 ? ne*(Zeff-1)/((Zimp*(Zimp-1))) : zero(ne)
    ni = max(ne - Zimp*nimp,zero(ne))

    lnΔ_be = lnΔ_ei(ne, Te*1e3, ni, Ti*1e3; mu=Ab, Z=Zb)
    Γ_be = (2*pi*ne*(e0^4)*Zb^2*lnΔ_be)/(m_b^2)

    electron_drag = ((8*Γ_be*m_b)/(3*sqrt(pi)*m_e*v_e^3))*v_b^3

    lnΔ_bi = lnΔ_ii(ni, Ti*1e3, ni, Ti*1e3, ne, Te*1e3, v_b/c0; mu1=Ab, Z1=Zb, mu2=Ai, Z2=Zi)
    Γ_bi = (2*pi*ni*(e0^4)*Zi^2*Zb^2*lnΔ_bi)/(m_b^2)

    lnΔ_bimp = lnΔ_ii(ni, Ti*1e3, nimp, Ti*1e3, ne, Te*1e3, v_b/c0; mu1=Ab, Z1=Zb, mu2=Aimp, Z2=Zimp)
    Γ_bimp = (2*pi*nimp*(e0^4)*(Zimp^2)*(Zb^2)*lnΔ_bimp)/(m_b^2)

    ion_drag = 2*m_b*(Γ_bi/m_i + Γ_bimp/m_imp)

    return electron_drag - ion_drag
end

function critical_energy(ne, Te, Ti, Zeff; Emax = 300.0,
                         Ai = H2_amu, Aimp = C6_amu, Ab=H2_amu,
                         Zi = 1, Zimp=6, Zb=1)
    # Calculates critical energy using root finding on drags
    # Energetic ion distribution resulting from neutral beam injection in tokamaks: J Gaffey 1976 J. Plasma Phys. 16 149
    # E0/Te in keV, ne in cm^-3

    Ec = find_zero(x -> _electron_ion_drag_difference(x, ne, Te, Ti, Zeff;
                   Ai=Ai, Aimp=Aimp, Ab=Ab, Zi=Zi,Zimp=Zimp, Zb=Zb),
                  (0, Emax))
    return Ec
end

function approx_critical_energy(ne, Te, Zeff; correction_factor = 1.0,
                                Ai = H2_amu, Aimp = C6_amu, Ab=H2_amu,
                                Zi = 1, Zimp=6, Zb=1)
    # Calculate critical energy assuming lnΔ_be == lnΔ_bi
    # For DIII-D a correction factor of (lnΔ_bi/lnΔ_be)^(2/3) ≈ 1.2 can be used
    # Heating of toroidal plasmas by neutral injection: T H Stix 1972 Plasma Phys. 14 367
    # E0/Te in keV, ne in cm^-3

    nimp = Zeff > 1 ? ne*(Zeff - 1)/(Zimp*(Zimp-1)) : zero(ne)
    ni = max(ne - Zimp*nimp,zero(ne))

    # average charge to mass ratio (avg_cmr) is where approximation comes in
    avg_cmr = (ni*(Zi^2/Ai) + nimp*(Zimp^2/Aimp))/ne
    Ec = 14.8 * Ab * Te * correction_factor*avg_cmr^(2/3)

    return Ec
end

function _slowing_down(E::T, E0::T, P::Vector, P0::Vector;
                       tau_on = 0.0, tau_off = 1.0, tau = 1.0,
                       Ai = H2_amu, Ab = Ai, Aimp = C6_amu,
                       Te = 1, Ti = Te, Zeff = 1, Zi=1, Zimp=6, Zb=Zi, ne = 1) where T<:Number
    # Te/Ti in keV; ne in cm^-3

    E >= E0  && return 0.0

    m_b = mass_u*Ab

    v = sqrt(2*(e0*E*1e3)/(m_b))
    v3 = v^3
    vb = sqrt(2*(e0*E0*1e3)/(m_b))
    vb3 = vb^3

    J = v*((e0*1e3)/m_b)

    tau_s = slowing_down_time(ne, Te, Ti, Zeff; Ai = Ai, Aimp=Aimp, Ab = Ab, Zi=Zi, Zimp=Zimp, Zb=Zb)
    Ec = critical_energy(ne, Te, Ti, Zeff; Ai = Ai, Aimp=Aimp, Ab = Ab, Zi=Zi, Zimp=Zimp, Zb=Zb)

    vc = sqrt(2*(e0*Ec*1e3)/m_b)
    vc3 = vc^3

    inv_v3vc3 = inv(v3 + vc3)
    u1 = v3/vb3
    u2 = (vb3 + vc3)*inv_v3vc3
    u = (u1*u2)^(Zeff/6)

    if tau_on > tau_off
        tau_off = Inf
    end
    t_b = tau_s*log((vb3+vc3)/vc3)/3
    t_0 = tau_on*t_b
    t_th = tau_s*log(u2)/3
    t_1 = tau_off*t_b - t_0
    t = tau*t_b - t_0

    S = min(t,t_b,t_1)/t_b
    U = heaviside(t - t_th) - heaviside(t - t_1 - t_th)
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
                       Ai = H2_amu, Ab = Ai, Aimp = C6_amu,
                       Te = 1, Ti = Te, Zeff = 1, Zi=1, Zimp=6, Zb=Zi, ne = 1) where T<:Number
    # Te/Ti in keV; ne in cm^-3

    E >= E0  && return 0.0

    m_b = mass_u*Ab
    m_i = mass_u*Ai

    v = sqrt(2*(e0*E*1e3)/(m_b))
    v3 = v^3
    vb = sqrt(2*(e0*E0*1e3)/(m_b))
    vb3 = vb^3

    J = v*((e0*1e3)/m_b)

    tau_s = slowing_down_time(ne, Te, Ti, Zeff; Ai = Ai, Aimp=Aimp, Ab = Ab, Zi=Zi, Zimp=Zimp, Zb=Zb)
    Ec = critical_energy(ne, Te, Ti, Zeff; Ai = Ai, Aimp=Aimp, Ab = Ab, Zi=Zi, Zimp=Zimp, Zb=Zb)

    vc = sqrt(2*(e0*Ec*1e3)/m_b)
    vc3 = vc^3

    t_b = tau_s*log((vb3+vc3)/vc3)/3

    beta = (m_i*Zeff)/(2*m_b)
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
                      vd_para=0.0, sigma=-1, Ai = H2_amu) where S<:Number

    m_i = mass_u*Ai

    T_perp = T_perp*1e3*e0
    T_para = T_para*1e3*e0
    Ed_para = (0.5*m_i*vd_para^2)

    E = energy*e0*1e3
    xi = clamp(sigma*pitch,nextfloat(-1.0),prevfloat(1.0))
    J = e0*1e3
    A = sqrt(E/(pi*T_para*T_perp^2))
    T1 = -(E*xi^2 + Ed_para - vd_para*xi*sqrt(2*m_i*E))/T_para
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
