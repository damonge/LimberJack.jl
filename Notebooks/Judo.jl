using Random
using ForwardDiff
using LimberJack
using NPZ
using FITSIO
using LinearAlgebra

des_data = npzread("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/cl_DESgc__2_DESwl__3.npz")["cl"]
des_data = transpose(des_data)[1:39]
des_ell = npzread("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/cl_DESgc__2_DESwl__3.npz")["ell"]
des_ell = [Int(floor(l)) for l in des_ell]
des_cov = npzread("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/cov_DESgc__2_DESwl__3_DESgc__2_DESwl__3.npz")["cov"]
des_cov = des_cov[1:39, 1:39]
des_cov = Symmetric(Hermitian(des_cov))
des_err = (view(des_cov, diagind(des_cov)))[1:39].^0.5
des_nzs = FITS("/mnt/zfsusers/jaimerz/PhD/LimberJack.jl/data/y1_redshift_distributions_v1.fits")
des_nz = read(des_nzs["nz_source_mcal"], "BIN2")
des_zs = read(des_nzs["nz_source_mcal"], "Z_MID")
des_inv_cov = inv(des_cov)

true_cosmology = LimberJack.Cosmology(0.25, 0.05, 0.67, 0.96, 0.81,
                                      tk_mode="EisHu", Pk_mode="Halofit");

nz = @. exp(-0.5*((true_cosmology.zs-0.5)/0.05)^2)
tg = NumberCountsTracer(true_cosmology, des_zs, des_nz, 2.)
ts = WeakLensingTracer(true_cosmology, des_zs, des_nz)
true_cl = [angularCℓ(true_cosmology, ts, tg, ℓ) for ℓ in des_ell]
cl_data = true_cl #+ des_err .* rand(length(true_cl));

function Xi_gs(Ωm::Real, s8::Real)
    cosmo = LimberJack.Cosmology(Ωm, 0.05, 0.67, 0.96, s8, 
                                 tk_mode="EisHu", Pk_mode="Halofit")
    tg = NumberCountsTracer(cosmo, des_zs, des_nz, 2.)
    ts = WeakLensingTracer(cosmo, des_zs, des_nz)
    Cℓ_gs = [angularCℓ(cosmo, tg, ts, ℓ) for ℓ in des_ell] 
    diff = cl_data.-Cℓ_gs
    Xi2 = dot(diff, des_inv_cov * diff)
    return Xi2
end

∂2f_∂x∂y(x, y) = ForwardDiff.derivative(y -> ForwardDiff.derivative(x -> Xi_gs(x, y), x), y) 

Wms = 0.104:0.004:0.5
s8s = 0.406:0.006:1.0

out = [[Xi_gs(Wm, s8) for s8 in s8s] for Wm in Wms]
out = reduce(vcat,transpose.(out))
outdiff = [[∂2f_∂x∂y(Wm, s8) for s8 in s8s] for Wm in Wms];
outdiff = reduce(vcat,transpose.(outdiff))

npzwrite("judo_comps.npz", Dict("Wms" => Wms, "s8s" => s8s, "Xi2" => out, "ddXi2" => outdiff))
