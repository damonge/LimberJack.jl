using Random
using ForwardDiff
using LimberJack
using NPZ
using LinearAlgebra
using LaTeXStrings

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