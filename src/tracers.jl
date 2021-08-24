
struct NumberCountsTracer{T<:Real}
    wint::AbstractInterpolation{T, 1}
    bias::T
    wnorm::T
end

NumberCountsTracer(z, wz, bias) = begin
    wint = LinearInterpolation(z, wz, extrapolation_bc=0)
    area = quadgk(wint, z[1], z[end], rtol=1E-5)[1]
    NumberCountsTracer(wint, bias, 1.0/area)
end
