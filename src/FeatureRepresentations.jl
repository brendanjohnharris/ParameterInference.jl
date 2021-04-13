using Catch22
using StatsBase



catch24Names = vcat(:mean, :standard_deviation, Catch22.featureNames...)
function catch24(𝐱::AbstractVector{Float64})
    𝐟 = vcat(StatsBase.mean(𝐱), StatsBase.std(𝐱), catch22(𝐱)...)
    Catch22.featureVector(𝐟, catch24Names)
end
catch24(X::AbstractArray{Float64, 2}) = Catch22.featureMatrix(mapslices(catch24, X, dims=[1]), catch24Names)
export catch24

catch23Names = setdiff(catch24Names, [:FC_LocalSimple_mean1_tauresrat])
catch21Names = setdiff(Catch22.featureNames, [:FC_LocalSimple_mean1_tauresrat])
function catch23(𝐱::AbstractVector{Float64}) # Hot fix
    𝐟 = vcat(StatsBase.mean(𝐱), StatsBase.std(𝐱), catch22(𝐱, catch21Names)...)
    Catch22.featureVector(𝐟, catch23Names)
end
catch23(X::AbstractArray{Float64, 2}) = Catch22.featureMatrix(mapslices(catch23, X, dims=[1]), catch23Names)
export catch23

# List of feature functions:
#   - catch22
#   - catch24
#   -.........
function featureRepresentation(X, featureFunc::Function)
    F = featureFunc(X)
end
export featureRepresentation