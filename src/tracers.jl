
struct NumberCountsTracer{T<:Real}
    wint::AbstractInterpolation{T, 1}
    bias::T
end

NumberCountsTracer(cosmo::Cosmology, z, pz, bias) = begin
    # OPT: here we only optimize to calculate the area.
    #      perhaps it'd be best to just use Simpsons.
    pz_int = LinearInterpolation(z, pz, extrapolation_bc=0)
    area = quadgk(pz_int, z[1], z[end], rtol=1E-5)[1]
    chi = cosmo.chi(z)
    hz = Hmpc(cosmo, z)
    w_arr = @. (pz*hz/area)
    wint = LinearInterpolation(chi, w_arr, extrapolation_bc=0)
    NumberCountsTracer(wint, bias)
end
