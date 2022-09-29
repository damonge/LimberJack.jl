"""
    Emulator()

Emulator structure.

Arguments:
- `alphas::Matrix` : emulator's linear model for each k-scale
- `hypers`::Matrix : hyperparameters for each k-scale
- `trans_cosmos::Vector{Array}` : transformed cosmological parameters
- `training_karr::Vector{Float}` : array of k-scales at which the emulator was trained
- `inv_chol_cosmos::Matrix` : inverse Cholesky decomposition of training matter power spectra
- `mean_cosmos::Vector{Float}` : mean training cosmology
- `mean_log_Pks::Vector{Float}` : mean of the log training matter power spectra
- `std_log_Pks::Vector{Float}` : standard deviation of the log training matter power spectrum

Returns:
- `emulator::Structure` : emulator structure.

"""
struct Emulator
    alphas
    hypers
    trans_cosmos
    training_karr
    inv_chol_cosmos
    mean_cosmos
    mean_log_Pks
    std_log_Pks
end

"""
    Emulator(; path="../emulator/files.npz")

Emulator structure constructure.
Loads emulator files into the `Emulator` structure.

Arguments:
- `path::String` : path to emulator files

Returns:
- `emulator::Structure` : emulator structure.

"""
Emulator(; path="../emulator/files.npz") = begin
    files = npzread(path)
    trans_cosmos = files["trans_cosmos"]
    training_karr = files["training_karr"]
    hypers = files["hypers"]
    alphas = files["alphas"]
    inv_chol_cosmos = files["inv_chol_cosmos"]
    mean_cosmos = files["mean_cosmos"]
    mean_log_Pks = files["mean_log_Pks"]
    std_log_Pks = files["std_log_Pks"]
    #Note: transpose python arrays
    Emulator(alphas, hypers, 
             trans_cosmos', training_karr,
             inv_chol_cosmos, mean_cosmos,
             mean_log_Pks, std_log_Pks)
end

function _x_transformation(emulator::Emulator, point)
    return emulator.inv_chol_cosmos * (point .- emulator.mean_cosmos)'
end

function _y_transformation(emulator::Emulator, point)
    return ((point .- emulator.mean_log_Pks) ./ emulator.std_log_Pks)'
end

function _inv_y_transformation(emulator::Emulator, point)
    return exp.(emulator.std_log_Pks .* point .+ emulator.mean_log_Pks)
end

function _get_kernel(arr1, arr2, hyper)
    arr1_w = @.(arr1/exp(hyper[2:6]))
    arr2_w = @.(arr2/exp(hyper[2:6]))
    
    # compute the pairwise distance
    term1 = sum(arr1_w.^2, dims=1)
    term2 = 2 * (arr1_w' * arr2_w)'
    term3 = sum(arr2_w.^2, dims=1)
    dist = @.(term1-term2+term3)
    # compute the kernel
    kernel = @.(exp(hyper[1])*exp(-0.5*dist))
    return kernel
end

"""
    get_emulated_log_pk0(cosmo::CosmoPar)

Given a `CosmoPar` instance, emulates the value of the primordial \
matter power spectrum at the training k-scales.

Arguments:
- `CosmoPar::Stucture` : cosmological paramters structure

Returns:
- `karr::Vector{Float}` : array of training k-scales
- `log_pk0s::Vector{Dual}` : array of emulated log matter power spectrum

"""
function get_emulated_log_pk0(cosmo::CosmoPar)
    emulator = Emulator()
    cosmotype, params = reparametrize(cosmo) 
    params_t = _x_transformation(emulator, params')
    
    nk = length(emulator.training_karr)
    pk0s_t = zeros(cosmotype, nk)
    for i in 1:nk
        kernel = _get_kernel(emulator.trans_cosmos, params_t, emulator.hypers[i, :])
        pk0s_t[i] = dot(vec(kernel), vec(emulator.alphas[i,:]))
    end
    pk0s = vec(_inv_y_transformation(emulator, pk0s_t))
    return emulator.training_karr, pk0s
end

"""
    reparametrize(cosmo::CosmoPar)

Given a `CosmoPar` instance, it transforms the parameters \
to the parametrization used to train the emulator. 

Arguments:
- `CosmoPar::Stucture` : cosmological paramters structure

Returns:
- `cosmotype::Type` : type of the parameter array
- `params::Vector{Dual}` : array of transformed cosmological parameters

"""
function reparametrize(cosmo::CosmoPar)
    Ωc = cosmo.Ωm - cosmo.Ωb 
    wc = Ωc*cosmo.h^2
    wb = Ωb*cosmo.h^2
    params = [wc, wb, 2.7, cosmo.n_s, cosmo.h]
    cosmotype = eltype(params)
    return cosmotype, params
end