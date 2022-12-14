# LimberJack.jl

## Core

```Core``` performs the main computations of ```LimberJack.jl```. 
When using ```LimberJack.jl```, the first step is to create an instance of the  ```Cosmology``` structure.
This is as easy as calling:

```julia
    using LimberJack
    cosmology = Cosmology()
```

This will generate the an instance of ```Cosmology``` given the vanilla $\Lambda$CDM cosmology of ```CCL ```.
```Cosmology()``` then computes the value of the comoving distance, the growth factor, the growth rate and matter power spectrum at an array of values and generates interpolators for said quantites. 
The user can acces the value of these interpolator at an arbitrary input using the public functions of the model.

```@docs
LimberJack.Settings
LimberJack.CosmoPar
LimberJack.Cosmology
LimberJack.Ez
LimberJack.Hmpc
LimberJack.comoving_radial_distance
LimberJack.growth_factor
LimberJack.lin_Pk
LimberJack.nonlin_Pk
```

## Emulator
```@docs
LimberJack.Emulator
LimberJack.get_emulated_log_pk0
```

## Halofit
```@docs
LimberJack.get_PKnonlin
```

## Spectra
```@docs
LimberJack.Cℓintegrand
LimberJack.angularCℓs
```

## Tracers
```@docs
LimberJack.NumberCountsTracer
LimberJack.WeakLensingTracer
LimberJack.CMBLensingTracer
```
