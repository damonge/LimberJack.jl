# make.jl
using Documenter, GaussianProcess

makedocs(sitename = "LimberJack.jl",
         modules = [GaussianProcess],
         pages = ["Home" => "index.md",
                  "API" => "api.md"])
         