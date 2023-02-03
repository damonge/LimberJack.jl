function _ρDE_z(z, w0=-1., wa=0.)
    return (1+z)^(3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * z /(1+z))
end

function _X_z(z, ΩM, w0, wa)
    return ΩM*((1+z)^3)/((1-ΩM)*_ρDE_z(z, w0, wa))
end

function _w_z(z, w0, wa)
    return w0+wa*z/(1+z)
end

function _growth!(du,u,p,a)
    ΩM = p[1]
    w0 = p[2]
    wa = p[3]
    z = 1.0 / a - 1.0
    G = u[1]
    dG = u[2]
    du[1] = dG
    du[2] = -(3.5-1.5*_w_z(z, w0, wa)/(1+_X_z(z, ΩM, w0, wa)))*dG/a-1.5*(1-_w_z(z, w0,wa))/(1+_X_z(z, ΩM, w0, wa))*G/(a^2)
end

function _growth_solver(cpar::CosmoPar; w0=-1.0, wa=0.0)
    u₀ = [1.0,0.0]
    aspan = (0.99e-3, 1.01)
    p = [cpar.Ωm, w0, wa]

    prob = ODEProblem(_growth!, u₀, aspan, p)
    sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-6)
    return sol
end

function _D_a(a::Array, sol::SciMLBase.ODESolution)
    [u for (u,t) in sol.(a)] .* a ./ (sol(1)[1,:])
end

function _f_a(a::Array, sol::SciMLBase.ODESolution)
    G = [u for (u,t) in sol.(a)]
    G_prime = [t for (u,t) in sol.(a)]
    D = G .* a
    D_prime = G_prime .* a .+ G
    return a ./ D .* D_prime
end

function get_growth(cpar::CosmoPar, settings::Settings; kwargs...)
    if settings.Dz_mode == "RK2"
        # ODE solution for growth factor
        x_Dz = LinRange(0, log(1+1100), settings.nz_pk)
        dx_Dz = x_Dz[2]-x_Dz[1]
        z_Dz = @.(exp(x_Dz) - 1)
        a_Dz = @.(1/(1+z_Dz))
        aa = reverse(a_Dz)
        e = _Ez(cpar, z_Dz)
        ee = reverse(e)
        dd = zeros(settings.cosmo_type, settings.nz_pk)
        yy = zeros(settings.cosmo_type, settings.nz_pk)
        dd[1] = aa[1]
        yy[1] = aa[1]^3*ee[end]
        
        for i in 1:(settings.nz_pk-1)
            A0 = -1.5 * cpar.Ωm / (aa[i]*ee[i])
            B0 = -1. / (aa[i]^2*ee[i])
            A1 = -1.5 * cpar.Ωm / (aa[i+1]*ee[i+1])
            B1 = -1. / (aa[i+1]^2*ee[i+1])
            yy[i+1] = (1+0.5*dx_Dz^2*A0*B0)*yy[i] + 0.5*(A0+A1)*dx_Dz*dd[i]
            dd[i+1] = 0.5*(B0+B1)*dx_Dz*yy[i] + (1+0.5*dx_Dz^2*A0*B0)*dd[i]
        end
        
        y = reverse(yy)
        d = reverse(dd)
        
        Dzi = linear_interpolation(z_Dz, d./d[1], extrapolation_bc=Line())
        fs8zi = linear_interpolation(z_Dz, -cpar.σ8 .* y./ (a_Dz.^2 .*e.*d[1]),
                                     extrapolation_bc=Line())

    elseif settings.Dz_mode == "OrdDiffEq"
        sol = _growth_solver(cpar)
        z_Dz = LinRange(0, 1100, settings.nz_pk)
        a_Dz = @.(1/(1+z_Dz))
        Dzs = _D_a(a_Dz, sol)
        Dzi = cubic_spline_interpolation(z_Dz, Dzs, extrapolation_bc=Line())
        fs8zs = (cpar.σ8 .* Dzs ./ Dzs[1]) .* _f_a(a_Dz, sol)
        fs8zi = cubic_spline_interpolation(z_Dz, fs8zs, extrapolation_bc=Line())
        
    elseif settings.Dz_mode == "Custom"
        zs_c, Dzs_c = kwargs[:Dz_custom]
        d = zs_c[2]-zs_c[1]
        Dzi = cubic_spline_interpolation(zs_c, Dzs_c, extrapolation_bc=Line())
        dDzs_mid = (Dzs_c[2:end].-Dzs_c[1:end-1])/d
        zs_mid = (zs_c[2:end].+zs_c[1:end-1])./2
        dDzi = linear_interpolation(zs_mid, dDzs_mid, extrapolation_bc=Line())
        dDzs_c = dDzi(zs_c)
        fs8zi = cubic_spline_interpolation(zs_c, -cpar.σ8 .* (1 .+ zs_c) .* dDzs_c,
                                           extrapolation_bc=Line())
    else
        println("Transfer function not implemented")
    end
        
    return Dzi, fs8zi
end