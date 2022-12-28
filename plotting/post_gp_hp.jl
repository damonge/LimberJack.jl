using LimberJack
using GaussianProcess
using CSV
using DataFrames
using NPZ
using FITSIO
using PythonCall
using Interpolations
np = pyimport("numpy")

function get_gp_hp(chain;
                fid_cosmo=LimberJack.Cosmology(),
                latent_x=Vector(0:0.3:3))
    ###
    nsamples = length(chain[!, "h"])
    mu = fid_cosmo.Dz(vec(latent_x))
    eta = chain[!, "eta"]
    l = chain[!, "l"]
    σ8 = 0.81 * ones(nsamples)

    nx = 11
    vs = zeros(Float64, nx, nsamples)
    for i in 1:nx
        vs[i, :] = chain[!, string("v[", i ,"]")]
    end
    ###
    latent_x = Vector(0:0.3:3)
    x = Vector(range(0., stop=3., length=100))
    d = x[2] - x[1]
    gps = similar(vs)
    fs8s = zeros(Float64, 100, nsamples)
    for i in 1:length(l)
        K = sqexp_cov_fn(latent_x; eta=eta[i], l=l[i])
        gp = latent_GP(mu, vs[:, i], K)
        gps[:, i] = gp
        gp_cond = conditional(latent_x, x, gp, sqexp_cov_fn;
                              eta=1.0, l=l[i])
        Dzi = linear_interpolation(x, gp_cond ./ gp_cond[1], extrapolation_bc=Line())
        dDzs_mid = (gp_cond[2:end].-gp_cond[1:end-1])/d
        zs_mid = (x[2:end].+x[1:end-1])./2
        dDzi = linear_interpolation(zs_mid, dDzs_mid, extrapolation_bc=Line())
        dDzs_c = dDzi(x)
        fs8 = -σ8[i] .* (1 .+ x) .* dDzs_c
        fs8s[:, i] = fs8
    end

    return gps, fs8s
end

function post_gp_hp(fol_path)
    i = 1
    while isfile(string(fol_path, "chain_", i, ".csv"))
        file_path = string(fol_path, "chain_", i, ".csv")
        chain = CSV.read(file_path, DataFrame)
        gp, fs8 = get_gp_hp(chain)
        println(i)
        npzwrite(string(fol_path, "gp_", i ,".npz"), gp)
        npzwrite(string(fol_path, "fs8_", i ,".npz"), fs8)
        i += 1
    end
end

post_gp_hp("ND_RSD_gp_hp_TAP_0.6/")