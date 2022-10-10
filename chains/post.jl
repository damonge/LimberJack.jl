using LimberJack
using GaussianProcess
using CSV
using DataFrames

function get_gp(chain;
                fid_cosmo=LimberJack.Cosmology(),
                latent_x=Vector(0:0.3:3))
    ###
    mu = fid_cosmo.Dz(vec(latent_x))
    eta = chain[!, "eta"]
    l = chain[!, "l"]
    vs = zeros(Float64, nx, length(l))
    for i in 1:nx
        vs[i, :] = chain[!, string("v[", i ,"]")]
    end
    ###
    
    gps = similar(vs)
    for i in 1:length(l)
        K = sqexp_cov_fn(latent_x; eta=eta[i], l=l[i])
        gps[:, i] = latent_GP(mu, vs[:, i], K)
    end
    
    return gps
end

function post(fol_path)
    i = 1 
    while isfile(string(fol_path, "chain_", i, ".csv"))
        file_path = string(fol_path, "chain_", i, ".csv")
        chain = CSV.read(file_path, DataFrame)
        gp = get_gp(chain)
        println(i)
        npzwrite(string(fol_path, "gp_", i ,".npz"), get_gp(chain))
        i += 1
    end    
end

post("ND_gp_hp_TAP_0.6/")