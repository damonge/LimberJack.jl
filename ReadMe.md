# LimberJack.jl

[![Build Status](https://github.com/JaimeRZP/LimberJack.jl/workflows/CI/badge.svg)](https://github.com/JaimeRZP/LimberJack.jl/actions?query=workflow%3ALimberJack-CI+branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jaimerzp.github.io/LimberJack.jl/dev/)

![](https://raw.githubusercontent.com/JaimeRZP/LimberJack.jl/main/docs/src/assets/LimberJack_logo.png)

A differentiable cosmological code in Julia

## Installation

Once you have installed ```Julia``` you can install ```LimberJack.jl``` by:
1. Clone the git repository
2. From the repository directory open ```Julia```
3. In the ```Julia``` command line run:
``` julia
    using Pkg
    Pkg.add(path=".")
```

## Use

``` julia
    using LimberJack
    cosmology = Cosmology()
    Dz = growth_factor(cosmology, z)
```

## Challenges

1. Threading parallalization 
2. GPU implementation