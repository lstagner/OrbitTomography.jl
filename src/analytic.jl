function _slowing_down(E::T, p::T, E0::T, p0::T;
                      th_amu = H2_amu, b_amu = th_amu,
                      Te = 1, Zeff = 1, ne = 1, ni = ne, Zi = 1) where T<:Number
    m_e = mass_u*e_amu
    m_b = mass_u*b_amu
    m_th = mass_u*th_amu

    tau_s = Te^1.5 # there are other factors as well
    E >= E0  && return 0.0
    v = sqrt(2*(e0*E*1e3)/(m_b))
    vb = sqrt(2*(e0*E0*1e3)/(m_b))
    vc3 = 0.75*sqrt(pi)*((2*(Te*e0*1e3)/m_e)^(1.5))*((ni/ne) * Zi^2 * (m_e/(m_th)))

    beta = (m_th*Zeff)/(2*m_b)
    alpha = (beta/3)*(1 - p0^2)*log((1 + vc3/(v^3))/(1+vc3/(vb^3)))
    g = tau_s*inv(v^3 + vc3)*inv(sqrt(4*pi*alpha))*exp(-((p - p0)^2)/(4*alpha))
end

function slowing_down(energy::AbstractVector, pitch::AbstractVector, E0, p0; kwargs...)

    nenergy = length(energy)
    npitch = length(pitch)

    f_slow = zeros(nenergy,npitch)
    for i=1:nenergy, j=1:npitch
        f_slow[i,j] = _slowing_down(energy[i],pitch[j], E0, p0; kwargs...)
    end

    N, err = hcubature(x->_slowing_down(x[1],x[2], E0, p0; kwargs...), (0.0,-1.0), (1.1*E0,1.0))
    return f_slow/N
end

function _bimaxwellian(energy::S, pitch::S, T_perp::S, T_para::S;
                      vd_para=0.0, sigma=-1, th_amu = H2_amu) where S<:Number

    m_th = mass_u*th_amu

    T_perp = T_perp*1e3*e0
    T_para = T_para*1e3*e0
    Ed_para = (0.5*m_th*vd_para^2)

    E = energy*e0*1e3
    xi = sigma*pitch

    A = sqrt(E/(pi*T_para*T_perp^2))
    T1 = -(E*xi^2 + Ed_para - vd_para*xi*sqrt(2*m_th*E))/T_para
    T2 = -(1-xi^2)*E/T_perp
    return A*exp(T1 + T2)
end

function bimaxwellian(energy::AbstractVector, pitch::AbstractVector, T_perp, T_para; kwargs...)
    nenergy = length(energy)
    npitch = length(pitch)
    f_ep = zeros(nenergy, npitch)
    for i=1:nenergy, j=1:npitch
        f_ep[i,j] = _bimaxwellian(energy[i], pitch[j], T_perp, T_para; kwargs...)
    end
    T = (T_para + 2*T_perp)/3
    N, err = hcubature(x->_bimaxwellian(x[1], x[2], T_perp, T_para; kwargs...), (0.0,-1.0), (10*T,1.0))
    return f_ep/N
end

function maxwellian(energy, pitch, T; kwargs...)
    bimaxwellian(energy, pitch, T, T; kwargs...)
end
