abstract type Tracer end

struct NumberCountsTracer <: Tracer
    warr
    chis
    wint::AbstractInterpolation
    bias
    lpre::Int
end

NumberCountsTracer(cosmo::Cosmology, z_n, nz;
                   bias=1.0) = begin

    nz_int = LinearInterpolation(z_n, nz, extrapolation_bc=0)
    
    res = cosmo.settings.nz
    z_w = range(0.00001, stop=z_n[end], length=res)
    dz_w = (z_w[end]-z_w[1])/res
    nz_w = nz_int(z_w)
    
    #nz_norm = sum(0.5 .* (nz_w[1:res-1] .+ nz_w[2:res]) .* dz_w)
    nz_norm = trapz(z_w, nz_w)
    
    chi = cosmo.chi(z_w)
    hz = Hmpc(cosmo, z_w)
    
    w_arr = @. (nz_w*hz/nz_norm)
    wint = LinearInterpolation(chi, w_arr, extrapolation_bc=0)
    
    NumberCountsTracer(w_arr, chi, wint, bias, 0)
end

struct WeakLensingTracer <: Tracer
    warr
    chis
    wint::AbstractInterpolation
    bias
    lpre::Int
end

WeakLensingTracer(cosmo::Cosmology, z_n, nz;
                  mbias=-1.0, IA_params=[0.0, 0.0]) = begin
    
    nz_int = LinearInterpolation(z_n, nz, extrapolation_bc=0)
    
    res = cosmo.settings.nz
    z_w = range(0.00001, stop=z_n[end], length=res)
    dz_w = (z_w[end]-z_w[1])/res
    nz_w = nz_int(z_w)
    chi = cosmo.chi(z_w)
    
    #nz_norm = sum(0.5 .* (nz_w[1:res-1] .+ nz_w[2:res]) .* dz_w)
    nz_norm = trapz(z_w, nz_w)

    # Calculate chis at which to precalculate the lensing kernel
    # OPT: perhaps we don't need to sample the lensing kernel
    #      at all zs.
    # Calculate integral at each chi
    w_itg(chii) = @.(nz_w*(1-chii/chi))
    #w_arr = [sum(@.(0.5*(w_itg(chi[i])[i:res-1]+w_itg(chi[i])[i+1:res])*dz_w))
    #         for i in 1:res]
    w_arr = zeros(typeof(cosmo.cosmo.Ωm), res)
    for i in 1:res
        w_arr[i] = trapz(z_w[i:res], w_itg(chi[i])[i:res])
    end
    
    # Normalize
    H0 = cosmo.cosmo.h/CLIGHT_HMPC
    lens_prefac = 1.5*cosmo.cosmo.Ωm*H0^2
    w_arr = @. w_arr * chi * lens_prefac * (1+z_w) / nz_norm
    
    if IA_params != [0.0, 0.0]
        hz = Hmpc(cosmo, z_w)
        As = get_IA(cosmo, z_w, IA_params)
        corr =  @. As * (nz_w * hz / nz_norm)
        w_arr = @. w_arr - corr
    end

    # Interpolate
    # Fix first element
    chi[1] = 0.0
    wint = LinearInterpolation(chi, w_arr, extrapolation_bc=0)
    bias = mbias+1.0 
    WeakLensingTracer(w_arr, chi, wint, bias , 2)
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

function get_IA(cosmo::Cosmology, zs, IA_params)
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
