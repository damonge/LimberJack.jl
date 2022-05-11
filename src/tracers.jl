abstract type Tracer end

struct NumberCountsTracer <: Tracer
    wint::AbstractInterpolation
    bias::Float64
    lpre::Int
end

NumberCountsTracer(cosmo::Cosmology, z_n::Vector{Float64}, nz::Vector{Float64};
                   bias=0.0) = begin
    # OPT: here we only integrate to calculate the area.
    #      perhaps it'd be best to just use Simpsons.
    nz_norm = trapz(z_n, nz)
    chi = cosmo.chi(z_n)
    hz = Hmpc(cosmo, z_n)
    w_arr = @. (nz*hz/nz_norm)
    wint = LinearInterpolation(chi, w_arr, extrapolation_bc=0)
    NumberCountsTracer(wint, bias, 0)
end

struct WeakLensingTracer <: Tracer
    wint::AbstractInterpolation
    bias::Float64
    lpre::Int
end

WeakLensingTracer(cosmo::Cosmology, z_n::Vector{Float64}, nz::Vector{Float64};
                  mbias=-1.0, IA_params=[0.0, 0.0]) = begin
    # N(z) normalization
    nz_norm = trapz(z_n, nz)
    nz_int = LinearInterpolation(z_n, nz, extrapolation_bc=0)

    # Calculate chis at which to precalculate the lensing kernel
    # OPT: perhaps we don't need to sample the lensing kernel
    #      at all zs.
    dz = (z_n[end]-z_n[1])/length(z_n)
    nchi = trunc(Int, z_n[end]/dz)
    z_w = range(0.00001, stop=z_n[end], length=nchi)
    chi = cosmo.chi(z_w)

    # Calculate integral at each chi
    w_itg(zz, chii) = nz_int(zz)*(1-chii/cosmo.chi(zz))
    # OPT: use simpsons/trapz?
    w_arr = [quadgk(zz -> w_itg(zz, chi[i]), z_w[i], z_n[end],
                    rtol=1E-4)[1]
             for i=1:nchi]
    # Normalize
    H0 = cosmo.cosmo.h/CLIGHT_HMPC
    lens_prefac = 1.5*cosmo.cosmo.Ωm*H0^2
    w_arr = @. w_arr * chi * lens_prefac * (1+z_w) / nz_norm
    
    if IA_params != [0.0, 0.0]
        hz = Hmpc(cosmo, z_w)
        As = get_IA(cosmo, z_w, IA_params)
        nz_w = [nz_int(z) for z in z_w]
        corr =  @. As * (nz_w * hz / nz_norm)
        w_arr = @. w_arr - corr
    end

    # Interpolate
    # Fix first element
    chi[1] = 0.0
    wint = LinearInterpolation(chi, w_arr, extrapolation_bc=0)
    bias = mbias+1 
    WeakLensingTracer(wint, bias , 2)
end

struct CMBLensingTracer <: Tracer
    wint::AbstractInterpolation
    bias::Float64
    lpre::Int
end

CMBLensingTracer(cosmo::Cosmology; nchi=100) = begin
    # chi array
    chis = range(0.0, stop=cosmo.chi_max, length=nchi)
    zs = cosmo.z_of_chi(chis)
    # Prefactor
    H0 = cosmo.cosmo.h/CLIGHT_HMPC
    lens_prefac = 1.5*cosmo.cosmo.Ωm*H0^2
    # Kernel
    w_arr = @. lens_prefac*chis*(1-chis/cosmo.chi_LSS)*(1+zs)

    # Interpolate
    wint = LinearInterpolation(chis, w_arr, extrapolation_bc=0)
    CMBLensingTracer(wint, 1.0, 1)
end

function get_IA(cosmo::Cosmology, zs::Vector{Float64}, IA_params::Vector{Float64})
    A_IA = IA_params[1]
    alpha_IA = IA_params[2]
    return @. A_IA*((1 + zs)/1.62)^alpha_IA * (0.0134 * cosmo.cosmo.Ωm / cosmo.Dz(zs))
end

function get_Fℓ(t::Tracer, ℓ::Real)
    if t.lpre == 2
        return @. sqrt((ℓ+2)*(ℓ+1)*ℓ*(ℓ-1))/(ℓ+0.5)^2
    elseif t.lpre == 1
        return @. (ℓ+1)*ℓ/(ℓ+0.5)^2
    else
        return 1
    end
end
