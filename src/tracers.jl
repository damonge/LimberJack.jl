abstract type Tracer end

"""
    NumberCountsTracer(warr, chis, wint, b, lpre)

Number counts tracer structure. 

Arguments:
- `warr::Vector{Dual}` : radial kernel array. 
- `chis::Vector{Dual}` : comoving distances array of the radial kernel.
- `wint::Interpolation` : interpolation of the radial kernel over comoving distance.
- `b::Dual`: matter-galaxy bias.
- `lpre::Int` : prefactor mark.

Returns:
- `NumberCountsTracer::NumberCountsTracer` : Number counts tracer structure.

"""
struct NumberCountsTracer <: Tracer
    warr
    chis
    wint::AbstractInterpolation
    b
end

"""
    NumberCountsTracer(cosmo::Cosmology, z_n, nz; kwargs...)

Number counts tracer structure constructor. 

Arguments:
- `cosmo::Cosmology` : cosmology structure. 
- `z_n::Vector{Dual}` : redshift array.
- `nz::Interpolation` : distribution of sources over redshift.

Kwargs:
- `b::Dual = 1` : matter-galaxy bias. 
 
Returns:
- `NumberCountsTracer::NumberCountsTracer` : Number counts tracer structure.

"""
NumberCountsTracer(cosmo::Cosmology, z_n, nz; kwargs...) = begin

    nz_int = LinearInterpolation(z_n, nz, extrapolation_bc=0)
    
    res = cosmo.settings.nz
    z_w = range(0.00001, stop=z_n[end], length=res)
    dz_w = (z_w[end]-z_w[1])/res
    nz_w = nz_int(z_w)
    nz_norm = trapz(z_w, nz_w)
    
    chi = cosmo.chi(z_w)
    hz = Hmpc(cosmo, z_w)
    
    w_arr = @. (nz_w*hz/nz_norm)
    wint = LinearInterpolation(chi, w_arr, extrapolation_bc=0)
    
    NumberCountsTracer(w_arr, chi, wint, kwargs[:b])
end

"""
    WeakLensingTracer(warr, chis, wint, b, lpre)

Weak lensing tracer structure. 
Arguments:

- `warr::Vector{Dual}` : radial kernel array. 
- `chis::Vector{Dual}` : comoving distances array of the radial kernel.
- `wint::Interpolation` : interpolation of the radial kernel over comoving distance.
- `b::Dual`: matter-galaxy bias.

Returns:
- `WeakLensingTracer::WeakLensingTracer` : Weak lensing tracer structure.

"""
struct WeakLensingTracer <: Tracer
    warr
    chis
    wint::AbstractInterpolation
    b
end

"""
    WeakLensingTracer(cosmo::Cosmology, z_n, nz; kwargs...)

Weak lensing tracer structure constructor. 

Arguments:

- `cosmo::Cosmology` : cosmology structure. 
- `z_n::Vector{Dual}` : redshift array.
- `nz::Interpolation` : distribution of sources over redshift.

Kwargs:
- `mb::Dual = 1` : multiplicative bias. 
- `IA_params::Vector{Dual} = [A_IA, alpha_IA]`: instrinsic aligment parameters.
 
Returns:
- `WeakLensingTracer::WeakLensingTracer` : Weak lensing tracer structure.

"""
WeakLensingTracer(cosmo::Cosmology, z_n, nz; kwargs...) = begin
    
    nz_int = LinearInterpolation(z_n, nz, extrapolation_bc=0)
    
    cosmo_type = cosmo.settings.cosmo_type
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
    w_arr = zeros(cosmo_type, res)
    @inbounds for i in 1:res
        w_arr[i] = trapz(z_w[i:res], w_itg(chi[i])[i:res])
    end
    
    # Normalize
    H0 = cosmo.cosmo.h/CLIGHT_HMPC
    lens_prefac = 1.5*cosmo.cosmo.Ωm*H0^2
    w_arr = @. w_arr * chi * lens_prefac * (1+z_w) / nz_norm
    
    if kwargs[:IA_params] != [0.0, 0.0]
        hz = Hmpc(cosmo, z_w)
        As = get_IA(cosmo, z_w, kwargs[:IA_params])
        corr =  @. As * (nz_w * hz / nz_norm)
        w_arr = @. w_arr - corr
    end

    # Interpolate
    # Fix first element
    chi[1] = 0.0
    wint = LinearInterpolation(chi, w_arr, extrapolation_bc=0)
    b = kwargs[:mb]+1.0 
    WeakLensingTracer(w_arr, chi, wint, b)
end

"""
    CMBLensingTracer(warr, chis, wint, lpre)

CMB lensing tracer structure. 

Arguments:

- `warr::Vector{Dual}` : radial kernel array. 
- `chis::Vector{Dual}` : comoving distances array of the radial kernel.
- `wint::Interpolation` : interpolation of the radial kernel over comoving distance.
- `lpre::Int` : prefactor mark.

Returns:
- `CMBLensingTracer::CMBLensingTracer` : CMB lensing tracer structure.

"""
struct CMBLensingTracer <: Tracer
    warr
    chis
    wint::AbstractInterpolation
end

"""
    CMBLensingTracer(cosmo::Cosmology; nchi=100)

CMB lensing tracer structure. 

Arguments:

- `cosmo::Cosmology` : cosmology structure.

Kwargs:
- `nchi::Int = 100` : number of nodes in the comoving distance array. 

Returns:
- `CMBLensingTracer::CMBLensingTracer` : CMB lensing tracer structure.

"""
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
    CMBLensingTracer(w_arr, chis, wint)
end

"""
    get_IA(cosmo::Cosmology, zs, IA_params)

CMB lensing tracer structure. 

Arguments:

- `cosmo::Cosmology` : cosmology structure.
- `zs::Vector{Dual}` : redshift array.
- `IA_params::Vector{Dual}` : Intrinsic aligment parameters.

Returns:
- `IA_corr::Vector{Dual}` : Intrinsic aligment correction to radial kernel.

"""
function get_IA(cosmo::Cosmology, zs, IA_params)
    A_IA = IA_params[1]
    alpha_IA = IA_params[2]
    return @. A_IA*((1 + zs)/1.62)^alpha_IA * (0.0134 * cosmo.cosmo.Ωm / cosmo.Dz(zs))
end
