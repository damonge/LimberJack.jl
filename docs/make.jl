# make.jl
using Documenter, LimberJack

makedocs(sitename = "LimberJack.jl",
         modules = [LimberJack],
         pages = ["Home" => "index.md",
                  "API" => "api.md"])
         
deploydocs(repo = "github.com/Damonge/LimberJack.jl")