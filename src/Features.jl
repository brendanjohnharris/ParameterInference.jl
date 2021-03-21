using StatsBase

# Nice and simple, function like type
Base.@kwdef struct Feature <: Function
    operation::Function = x -> x
    ID::UInt = rand(UInt)
    name::Symbol = Symbol(ID)
    # any metadata
end
(F::Feature)(x::Vector) = F.operation(x)

# Some custom features
TS_mean = Feature(operation=StatsBase.mean, name=:TS_mean)

TS_standard_deviation = Feature(operation=StatsBase.std, name=:TS_standard_deviation)



# struct FeatureSet <: Function
#     operations::Vector{Function} = [x -> x,]
# end