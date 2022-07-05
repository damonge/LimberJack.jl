function x_transformation(point, x_arr)
    """Pre-whiten the input parameters.

    Args:
        point (torch.tensor): the input parameters.

    Returns:
        torch.tensor: the pre-whitened parameters.
    """

    N, M = size(x_arr)
    cov_train = cov(x_arr)
    chol_train = cholesky(cov_train).U'
    mean_train = mean(x_arr, dims=1)
    # calculate the transformed training points
    transformed = inv(chol_train) * (point .- mean_train)'

    return transformed
end

function y_transformation(point, y_arr)
    """Transform the outputs.

    Args:
        yvalues (np.ndarray): the values to be transformed

    Returns:
        np.ndarray: the transformed outputs
    """
    ylog = log.(y_arr)
    ymean = mean(ylog, dims=1)
    ystd = std(ylog, dims=1)
    return ((point .- ymean) ./ ystd)'
end

function inv_y_transformation(point, ylog)
    """Transform the outputs.

    Args:
        yvalues (np.ndarray): the values to be transformed

    Returns:
        np.ndarray: the transformed outputs
    """
    
    ymean = mean(ylog, dims=1)
    ystd = std(ylog, dims=1)
    return @.(exp(ylog * ystd + ymean))
end

function get_kernel(arr1, arr2, hyper)
    """Compute the kernel matrix between the sets of training cosmoligical parameters
       and the desired set of cosmological parameters.
    Args:
        x1 (np.ndarray): [N x d] tensor of points.
        x2 (np.ndarray): [M x d] tensor of points.
        hyper (np.ndarray): [d+1] tensor of hyperparameters.
    Returns:
        np.ndarray: a tensor of size [N x M] containing the kernel matrix.
    """
    
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

function get_Pk(params_input, x_input, y_input, alphas)
    params = x_transformation(params_input)
    x_train = x_transformation(x_input)
    y_train = x_transformation(y_input)
    
    preds = []
    for i in 1:40
        kernel = get_kernel(xtrain, param, hyper[i, :])
        mean = dot(vec(kernel), vec(alphas[i,:]))
        push!(preds, mean)
    end
    return vec(inv_y_transformation(preds, y_train))
end
