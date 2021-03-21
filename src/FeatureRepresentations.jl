using Catch22
using StatsBase



catch24Names = vcat(:mean, :standard_deviation, Catch22.featureNames...)
function catch24(𝐱::AbstractVector{Float64})
    𝐟 = vcat(StatsBase.mean(𝐱), StatsBase.std(𝐱), catch22(𝐱)...)
    Catch22.featureVector(𝐟, catch24Names)
end
catch24(X::AbstractArray{Float64, 2}) = Catch22.featureMatrix(mapslices(catch24, X, dims=[1]), catch24Names)
export catch24

# List of feature functions:
#   - catch22
#   - catch24
#   -.........
function featureRepresentation(X, featureFunc::Function)
    F = featureFunc(X)
end
export featureRepresentation