struct Emulator
    alphas
    hypers
    training_cosmos
    training_lin_Pks
    training_karr
    inv_chol_cosmos
    mean_cosmos
    mean_log_Pks
    std_log_Pks
end

Emulator() = begin
    training_cosmos = npzread("../emulator_files/xinputs.npz")["arr_0"]
    training_lin_Pks = npzread("../emulator_files/yinputs.npz")["arr_0"]
    training_karr = npzread("../emulator_files/k_arr.npz")["arr_0"]
    hypers = npzread("../emulator_files/hypers.npz")["arr_0"]
    alphas = npzread("../emulator_files/alphas.npz")["arr_0"]
    
    cov_cosmos = cov(training_cosmos)
    chol_cosmos = cholesky(cov_cosmos).U'
    inv_chol_cosmos = inv(chol_cosmos)
    mean_cosmos = mean(training_cosmos, dims=1)
    
    log_Pks = log.(training_lin_Pks)
    mean_log_Pks = mean(log_Pks, dims=1)
    std_log_Pks = std(log_Pks, dims=1)
    Emulator(alphas, hypers, 
             training_cosmos, training_lin_Pks, training_karr,
             inv_chol_cosmos, mean_cosmos,
             mean_log_Pks, std_log_Pks)
end

function x_transformation(emulator::Emulator, point)
    return emulator.inv_chol_cosmos * (point .- emulator.mean_cosmos)'
end

function y_transformation(emulator::Emulator, point)
    return ((point .- emulator.mean_log_Pks) ./ emulator.std_log_Pks)'
end

function inv_y_transformation(emulator::Emulator, point)
    return exp.(emulator.std_log_Pks .* point' .+ emulator.mean_log_Pks)
end

function get_kernel(arr1, arr2, hyper)
    arr1_w = @.(arr1/exp(hyper[2:6]))
    arr2_w = @.(arr2/exp(hyper[2:6]))
    
    # compute the pairwise distance
    term1 = sum(arr1_w.^2, dims=1)
    term2 = 2 * arr1_w' * arr2_w
    term3 = sum(arr2_w.^2, dims=1)'
    dist = term1 - term2' .+ term3

    # compute the kernel
    kernel = @.(exp(hyper[1]) * exp(-0.5 * dist))

    return kernel
end

function get_emulated_log_pk0(cosmo::CosmoPar)
    emulator = Emulator()
    cosmotype, params = reparametrize(cosmo) 
    params_t = x_transformation(emulator, params')
    x_t = x_transformation(emulator, emulator.training_cosmos)
    y_t = y_transformation(emulator, emulator.training_lin_Pks)
    
    nk = length(emulator.training_karr)
    pk0s_t = zeros(cosmotype, nk)
    for i in 1:nk
        kernel = get_kernel(x_t, params_t, emulator.hypers[i, :])
        pk0s_t[i] = dot(vec(kernel), vec(emulator.alphas[i,:]))
    end
    pk0s = vec(inv_y_transformation(emulator, pk0s_t))
    return emulator.training_karr, pk0s
end

function reparametrize(cosmo::CosmoPar)
    Ωc = cosmo.Ωm - cosmo.Ωb 
    wc = Ωc*cosmo.h^2
    wb = cosmo.Ωb*cosmo.h^2
    params = [wc, wb, 2.7, cosmo.n_s, cosmo.h]
    cosmotype = eltype(params)
    return cosmotype, params
end