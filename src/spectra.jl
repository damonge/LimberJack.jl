
function Cℓintegrand(cosmo, t1, t2, logk, ℓ)
    k = exp(logk)
    chi = (ℓ+0.5)/k
    if chi > cosmo.chi_max
        return 0
    end
    z = cosmo.z_of_chi(chi)
    hz = Hmpc(cosmo, z)
    w1 = t1.wint(z)*t1.wnorm*hz*t1.bias
    w2 = t2.wint(z)*t2.wnorm*hz*t1.bias
    pk = power_spectrum(cosmo, k)
    k*w1*w2*pk
end

function angularCℓ(cosmo, t1, t2, ℓ)
    quadgk(lk -> Cℓintegrand(cosmo, t1, t2, lk, ℓ),
           log(10^-3), log(10^1))[1]/(ℓ+0.5)
end
