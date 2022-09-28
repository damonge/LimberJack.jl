# c/(100 km/s/Mpc) in Mpc
const CLIGHT_HMPC = 2997.92458

function _w_tophat(x::Real)
    x2 = x^2

    if x < 0.1
        w = 1. + x2*(-1.0/10.0 + x2*(1.0/280.0 +
            x2*(-1.0/15120.0 + x2*(1.0/1330560.0 +
            x2* (-1.0/172972800.0)))));
    else
        w = 3 * (sin(x)-x*cos(x))/(x2*x)
    end
    return w
end    

function _σR2(ks, pk, dlogk, R)
    x = ks .* R
    wk = _w_tophat.(x)
    integ = @. pk * wk^2 * ks^3
    # OPT: proper integration instead?
    return sum(integ)*dlogk/(2*pi^2)
end

"""
    Settings(cosmo_type, nz, nz_pk, nk, tk_mode, pk_mode, custom_Dz)

Cosmology constructor settings structure. 

Arguments:

- `cosmo_type::Type` : type of cosmological parameters. 
- `nz::Int` : number of nodes in the general redshift array.
- `nz_pk::Int` : number of nodes in the redshift array used for matter power spectrum.
- `nk::Int`: number of nodes in the matter power spectrum.
- `tk_mode::String` : choice of transfer function.
- `Pk_mode::String` : choice of matter power spectrum.
- `custom_Dz::Any` : custom growth factor.

Returns:

- `Settings` : cosmology settings.

"""
mutable struct Settings
    cosmo_type::DataType
    nz::Int
    nz_pk::Int
    nk::Int
    Pk_mode::String
    custom_Dz
end

"""
    CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB, Ωr, ΩΛ)

Cosmology parameters structure.  

Arguments:

- `Ωm::Dual` : cosmological matter density. 
- `Ωb::Dual` : cosmological baryonic density.
- `h::Dual` : reduced Hubble parameter.
- `n_s::Dual` : spectral index.
- `σ8::Dual`: variance of the matter density field in a sphere of 8 Mpc.
- `θCMB::Dual` : CMB temperature.
- `Ωr::Dual` : cosmological radiation density.
- `ΩΛ::Dual` : cosmological dark energy density.

Returns:

- `CosmoPar` : cosmology parameters structure.

"""
struct CosmoPar{T}
    Ωm::T
    Ωb::T
    h::T
    n_s::T
    σ8::T
    θCMB::T
    Ωr::T
    ΩΛ::T
end

"""
    CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB)

Cosmology parameters structure constructor.  

Arguments:
- `Ωm::Dual` : cosmological matter density. 
- `Ωb::Dual` : cosmological baryonic density.
- `h::Dual` : reduced Hubble parameter.
- `n_s::Dual` : spectral index.
- `σ8::Dual`: variance of the matter density field in a sphere of 8 Mpc.
- `θCMB::Dual` : CMB temperature.

Returns:
- `CosmoPar` : cosmology parameters structure.

"""
CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB) = begin
    # This is 4*sigma_SB*(2.7 K)^4/rho_crit(h=1)
    prefac = 2.38163816E-5
    Neff = 3.046
    f_rel = 1.0 + Neff * (7.0/8.0) * (4.0/11.0)^(4.0/3.0)
    Ωr = prefac*f_rel*θCMB^4/h^2
    ΩΛ = 1-Ωm-Ωr
    CosmoPar{Real}(Ωm, Ωb, h, n_s, σ8, θCMB, Ωr, ΩΛ)
end

"""
    Cosmology(Settings, CosmoPar,
              ks, pk0, logk, dlogk,
              zs, chi, z_of_chi, chi_max, chi_LSS, Dz, PkLz0, Pk)

Base cosmology structure.  

Arguments:
- `Settings::MutableStructure` : cosmology constructure settings. 
- `CosmoPar::Structure` : cosmological parameters.
- `ks::Dual` : scales array.
- `pk0::Dual`: primordial matter power spectrum.
- `logk::Dual` : log scales array.
- `dlogk::Dual` : increment in log scales.
- `zs::Dual` : redshift array.
- `chi::Dual` : comoving distance array.
- `z_of_chi::Dual` : redshift of comoving distance array.
- `chi_max::Dual` : upper bound of comoving distance array.
- `chi_LSS::Dual` : comoving distance to suface of last scattering.
- `Dz::Dual` : growth factor.
- `PkLz0::Dual` : interpolator of log primordial power spectrum over k-scales.
- `Pk::Dual` : matter power spectrum.

Returns:
- `CosmoPar` : cosmology parameters structure.

"""
struct Cosmology
    settings::Settings
    cosmo::CosmoPar
    # Redshift and background
    chi::AbstractInterpolation
    z_of_chi::AbstractInterpolation
    chi_max
    chi_LSS
    Dz::AbstractInterpolation
    PkLz0::AbstractInterpolation
    Pk::AbstractInterpolation
end

"""
    Cosmology(cpar::CosmoPar, settings::Settings)

Base cosmology structure constructor.

Calculates the LCDM expansion history based on the different \
species densities provided in `CosmoPar`.

The comoving distance is then calculated integrating the \
expansion history. 

Depending on the choice of transfer function in the settings, \
the primordial power spectrum is calculated using: 
- `tk_mode = "BBKS"` : the BBKS fitting formula (https://ui.adsabs.harvard.edu/abs/1986ApJ...304...15B)
- `tk_mode = "EisHu"` : the Eisenstein & Hu formula (arXiv:astro-ph/9710252)
- `tk_mode = "emulator"` : the Mootoovaloo et al 2021 emulator (arXiv:2105.02256v2)

is `custom_Dz = nothing`, the growth factor is obtained either by solving the Jeans equation. \
Otherwise, provided custom growth factor is used.


Depending on the choice of power spectrum mode in the settings, \
the matter power spectrum is either: 
- `Pk_mode = "linear"` : the linear matter power spectrum.
- `Pk_mode = "halofit"` : the Halofit non-linear matter power spectrum (arXiv:astro-ph/0207664).

Arguments:
- `Settings::MutableStructure` : cosmology constructure settings. 
- `CosmoPar::Structure` : cosmological parameters.

Returns:

- `Cosmology` : cosmology structure.

"""
Cosmology(cpar::CosmoPar, settings::Settings) = begin
    # Load settings
    cosmo_type = settings.cosmo_type
    nz_pk = settings.nz_pk
    nz = settings.nz
    nk = settings.nk
    zs_pk = range(0., stop=3., length=nz_pk)
    logk = range(log(0.0001), stop=log(100.0), length=nk)
    ks = exp.(logk)

    # Compute linear power spectrum at z=0.
    ks_emul, pk0_emul = get_emulated_log_pk0(cpar)
    pki_emul = LinearInterpolation(log.(ks_emul), log.(pk0_emul),
                                   extrapolation_bc=Line())
    #Renormalize Pk
    #σ8_2_here = _σR2(ks_emul, pk0_emul, dlogk, 8.0/cpar.h)
    #norm = cpar.σ8^2 / σ8_2_here
    #pk0_emul *= norm
    
    # OPT: interpolation method
    pki = LinearInterpolation(log.(ks_emul), log.(pk0_emul),
                              extrapolation_bc=Line())

    # Compute redshift-distance relation
    zs = range(0., stop=3., length=nz)
    norm = CLIGHT_HMPC / cpar.h
    
    chis = zeros(cosmo_type, nz)
    #OPT: Use cumul_integral
    for i in 1:nz
        zz = zs[i]
        chis[i] = quadgk(z -> 1.0/_Ez(cpar, z), 0.0, zz, rtol=1E-5)[1] * norm
    end
    # OPT: tolerances, interpolation method
    chii = LinearInterpolation(zs, chis, extrapolation_bc=Line())
    zi = LinearInterpolation(chis, zs, extrapolation_bc=Line())
    # Distance to LSS
    chi_LSS = quadgk(z -> 1.0/_Ez(cpar, z), 0.0, 1100., rtol=1E-5)[1] * norm

    if settings.custom_Dz == nothing
        # ODE solution for growth factor
        z_ini = 1000.0
        a_ini = 1.0/(1.0+z_ini)
        ez_ini = _Ez(cpar, z_ini)
        d0 = [a_ini^3*ez_ini, a_ini]
        a_s = reverse(@. 1.0 / (1.0 + zs))
        prob = ODEProblem(_dgrowth!, d0, (a_ini, 1.0), cpar)
        sol = solve(prob, Tsit5(), reltol=1E-6,
                    abstol=1E-8, saveat=a_s)
        # OPT: interpolation (see below), ODE method, tolerances
        # Note that sol already includes some kind of interpolation,
        # so it may be possible to optimize this by just using
        # sol directly.
        s = vcat(sol.u'...)
        Dzs_sol = reverse(s[:, 2] / s[end, 2])
        Dzi = LinearInterpolation(zs, Dzs_sol, extrapolation_bc=Line())
        Dzs = Dzi(zs_pk)
    else
        Dzs = custom_Dz
        Dzi = LinearInterpolation(zs_pk, Dzs, extrapolation_bc=Line())
    end

    if settings.Pk_mode == "linear"
        #OPT: easily vectorized
        Pks = [@. pk*Dzs^2 for pk in pk0_emul]
        Pks = reduce(vcat, transpose.(Pks))
        Pk = LinearInterpolation((log.(ks_emul), zs_pk), log.(Pks),
                                 extrapolation_bc=Line())
    elseif settings.Pk_mode == "Halofit"
        Pk = get_PKnonlin(cpar, zs_pk, ks, pki(logk), Dzs, cosmo_type)
    else
        print("Pk mode not implemented. Using linear Pk.")
    end
    Cosmology(settings, cpar,
              chii, zi, chis[end], chi_LSS,
              Dzi, pki, Pk)
end

"""
    Cosmology(Ωm, Ωb, h, n_s, σ8;
              θCMB=2.725/2.7, nk=500, nz=500, nz_pk=100,
              tk_mode="BBKS", Pk_mode="linear", custom_Dz=nothing)

Simple cosmology structure constructor that calls the base constructure.
Fills the `CosmoPar` and `Settings` structure based on the given parameters.

Arguments:

- `Ωm::Dual` : cosmological matter density. 
- `Ωb::Dual` : cosmological baryonic density.
- `h::Dual` : reduced Hubble parameter.
- `n_s::Dual` : spectral index.
- `σ8::Dual`: variance of the matter density field in a sphere of 8 Mpc.

Kwargs:

- `θCMB::Dual` : CMB temperature.
- `nz::Int` : number of nodes in the general redshift array.
- `nz_pk::Int` : number of nodes in the redshift array used for matter power spectrum.
- `nk::Int`: number of nodes in the matter power spectrum.
- `tk_mode::String` : choice of transfer function.
- `Pk_mode::String` : choice of matter power spectrum.
- `custom_Dz::Any` : custom growth factor.

Returns:

- `Cosmology` : cosmology structure.

"""
Cosmology(Ωm, Ωb, h, n_s, σ8; 
          θCMB=2.725/2.7, nz=500, nz_pk=100, nk=500,
          Pk_mode="linear", custom_Dz=nothing) = begin
    cosmo_type = eltype([Ωm, Ωb, h, n_s, σ8, θCMB])
    cpar = CosmoPar(Ωm, Ωb, h, n_s, σ8, θCMB)
    settings = Settings(cosmo_type, nz, nz_pk, nk, Pk_mode, custom_Dz)
    Cosmology(cpar, settings)
end

"""
    Cosmology()

Calls the simple Cosmology structure constructor for the \
paramters `[0.30, 0.05, 0.67, 0.96, 0.81]`.

Returns:

- `Cosmology` : cosmology structure.

"""
Cosmology() = Cosmology(0.30, 0.05, 0.67, 0.96, 0.81)

function σR2(cosmo::Cosmology, R)
    return _σR2(cosmo.ks, cosmo.pk0, cosmo.dlogk, R)
end

function _Ez(cosmo::CosmoPar, z)
    @. sqrt(cosmo.Ωm*(1+z)^3+cosmo.Ωr*(1+z)^4+cosmo.ΩΛ)
end

function _dgrowth!(dd, d, cosmo::CosmoPar, a)
    ez = _Ez(cosmo, 1.0/a-1.0)
    dd[1] = d[2] * 1.5 * cosmo.Ωm / (a^2*ez)
    dd[2] = d[1] / (a^3*ez)
end

"""
    chi_to_z(cosmo::Cosmology, chi)

Given a `Cosmology` instance, converts from comoving distance to redshift.  

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `chi::Dual` : comoving distance

Returns:
- `z::Dual` : redshift

"""
function chi_to_z(cosmo::Cosmology, chi)
    closest_chi, idx = findmin(abs(chi-cosmo.chis))
    if closest_chi >= chi
        idx += -1
    end

    return z
end

"""
    Ez(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the expansion rate (H(z)/H0). 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `Ez::Dual` : expansion rate 

"""
Ez(cosmo::Cosmology, z) = _Ez(cosmo.cosmo, z)

"""
    Hmpc(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the expansion history (H(z)) in Mpc. 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `Hmpc::Dual` : expansion rate 

"""
Hmpc(cosmo::Cosmology, z) = cosmo.cosmo.h*Ez(cosmo, z)/CLIGHT_HMPC

"""
    comoving_radial_distance(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the comoving radial distance. 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `z::Dual` : redshift

Returns:
- `Chi::Dual` : comoving radial distance

"""
comoving_radial_distance(cosmo::Cosmology, z) = cosmo.chi(z)

"""
    growth_factor(cosmo::Cosmology, z)

Given a `Cosmology` instance, it returns the growth factor (D(z) = log(δ)). 

Arguments:
- `cosmo::Cosmology` : cosmological structure
- `z::Dual` : redshift

Returns:
- `Chi::Dual` : comoving radial distance

"""
growth_factor(cosmo::Cosmology, z) = cosmo.Dz(z)

"""
    nonlin_Pk(cosmo::Cosmology, k, z)

Given a `Cosmology` instance, it returns the non-linear matter power spectrum (P(k,z)) \
using the Halofit fitting formula (arXiv:astro-ph/0207664). 

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `k::Dual` : scale
- `z::Dual` : redshift

Returns:
- `Pk::Dual` : non-linear matter power spectrum

"""
function nonlin_Pk(cosmo::Cosmology, k, z)
    return @. exp(cosmo.Pk(log(k), z))
end

"""
    lin_Pk(cosmo::Cosmology, k, z)

Given a `Cosmology` instance, it returns the linear matter power spectrum (P(k,z))

Arguments:
- `cosmo::Cosmology` : cosmology structure
- `k::Dual` : scale
- `z::Dual` : redshift

Returns:
- `Pk::Dual` : linear matter power spectrum

"""
function lin_Pk(cosmo::Cosmology, k, z)
    pk0 = @. exp(cosmo.PkLz0(log(k)))
    Dz2 = cosmo.Dz(z)^2
    return pk0 .* Dz2
end
