
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

function growth_solver(cpar::CosmoPar; w0=-1.0, wa=0.0)
    u₀ = [1.0,0.0]
    aspan = (0.99e-3, 1.01)
    p = [cpar.Ωm, w0, wa]

    prob = ODEProblem(_growth!, u₀, aspan, p)
    sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-6)
    return sol
end

function _D_z(z::Array, sol::SciMLBase.ODESolution)
    [u for (u,t) in sol.(_a_z.(z))] .* _a_z.(z) ./ (sol(_a_z(0.))[1,:])
end

function _D_z(z, sol::SciMLBase.ODESolution)
    return (_a_z(z) .* sol(_a_z(z))[1,:]/sol(_a_z(0.))[1,:])[1,1]
end

function _D_z(z, ΩM, w0, wa)
    sol = growth_solver(ΩM, w0, wa)
    return _D_z(z, sol)
end

function _f_a(a, sol::SciMLBase.ODESolution)
    G, G_prime = sol(a)
    D = G * a
    D_prime = G_prime * a + G
    return a / D * D_prime
end

function _f_a(a::Array, sol::SciMLBase.ODESolution)
    G = [u for (u,t) in sol.(a)]
    G_prime = [t for (u,t) in sol.(a)]
    D = G .* a
    D_prime = G_prime .* a .+ G
    return a ./ D .* D_prime
end

function _f_z(z, sol::SciMLBase.ODESolution)
    a = _a_z.(z)
    return _f_a(a, sol)
end

function _f_z(z, ΩM, w0, wa)
    sol = growth_solver(ΩM, w0, wa)
    return _f_z(z, sol)
end
