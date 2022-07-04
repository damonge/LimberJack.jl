function interpolate(x_old, y_old)
    """Function to interpolate the power spectrum along the redshift axis
    Args:
        inputs (list): x values, y values and new values of x
    Returns:
        np.ndarray: an array of the interpolated power spectra
    """

    itp = LinearInterpolation(log.(x_old), log.(y_old))
    
    return (y_new) -> let
        return itp(log.(y_new))
    end
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
    
    arr1_w = @.(arr1/exp(hyper[2:]))
    arr2_w = @.(arr2/exp(hyper[2:]))
    
    # compute the pairwise distance
    term1 = @.(sum(exp(arr1_w^2), 1))
    term2 = 2 * (arr1 * arr2)
    term3 = @.(sum(exp(arr2^2), 1))
    dist = term1 - term2 + term3'
    
    # compute the kernel
    kernel = np.exp(hyper[1]) * np.exp(-0.5 * dist)

    return kernel
end


function emulated_Pk(testpoint)
    """Compute the mean prediction of the GP.

    Args:
        testpoint (np.ndarray): the test points.

    Returns:
        np.ndarray: the mean prediction.
    """

    testpoint = x_transformation(testpoint)
    preds = []

    for i in 1:ngps
        kernel = get_kernel(xtrain, testpoint, hyper)
        mean = kernel .* alphas[i]
        push!(preds, mean)

    preds = np.array(preds) 
    preds = inv_y_transformation(preds)

    return preds
end

function x_transformation(x_arr)
    """Pre-whiten the input parameters.

    Args:
        point (torch.tensor): the input parameters.

    Returns:
        torch.tensor: the pre-whitened parameters.
    """

    N, M = size(x_arr)
    cov_train = cov(x_arr)
    chol_train = Cholesky(cov_train)
    mean_train = mean(x_arr, 1)

    # calculate the transformed training points
    transformed = inv(self.chol_train) .* (point .- mean_train)

    return transformed
end

function y_transformation(y_arr)
    """Transform the outputs.

    Args:
        yvalues (np.ndarray): the values to be transformed

    Returns:
        np.ndarray: the transformed outputs
    """
    ylog = log.(y_arr)
    ymean = mean(ylog, axis=0)
    ystd = std(ylog, axis=0)
    return (ylog - ymean) / ystd
end

function inv_y_transformation(ylog)
    """Transform the outputs.

    Args:
        yvalues (np.ndarray): the values to be transformed

    Returns:
        np.ndarray: the transformed outputs
    """
    
    ymean = mean(ylog, axis=0)
    ystd = std(ylog, axis=0)
    return @.(exp(ylog * ystd + ymean))
end
