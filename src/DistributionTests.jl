using HypothesisTests
using TensorCast


# Give two feature matrices, a test function (from HypothesisTests.jl) and calculate the test statistics for each feature common to the two feature matrices
function testFeatureDistributions(𝐟₁::AbstractVector, 𝐟₂::AbstractVector, test=TailedTestOfVariance; tail=:both)
    if tail ∈ (:firstgreater, :secondsmaller, :firstlarger)
        tail = :right
    elseif tail ∈ (:firstsmaller, :secondgreater, :secondlarger)
        tail = :left
    end
    p = pvalue(test(𝐟₁, 𝐟₂); tail=tail)
end
function testFeatureDistributions(F₁::AbstractArray, F₂::AbstractArray, test=TailedTestOfVariance; tail=:both)
    @cast p[i] := testFeatureDistributions(F₁[i, :], F₂[i, :], test; tail=tail)
end
function testFeatureDistributions(F₁::AbstractFeatureArray, F₂::AbstractFeatureArray, test=TailedTestOfVariance; tail=:both)
    # Intersect features before testing. In this case the test statistic returned refers to the order of the first input array. NaN means F₂ did not have a feature in F₁
    F̂₁, F̂₂ = intersectFeatures(F₁, F₂)
    p = testFeatureDistributions(Array(F̂₁), Array(F̂₂), test; tail=tail)
    F̂₁, p = intersectFeatures(F̂₁, p) # label p with the features of F̂₁
    fs = Catch22.featureDims(F̂₁)
    outp = deepcopy(F̂₁[:, 1])
    outp[:] .= NaN
    outp[fs] .= p[fs]
end
export testFeatureDistributions

function TailedTestOfVariance(x::AbstractVector{T}, y::AbstractVector{T}; scorediff=abs, statistic=mean) where {T<:Real}
    zx, zy = scorediff.(x .- statistic(x)), scorediff.(y .- statistic(y))
    EqualVarianceTTest(zx, zy)
end
export TailedTestOfVariance
function RobustTailedTestOfVariance(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    zx, zy = abs.(x .- statistic(x)), scorediff.(y .- median(y))
    EqualVarianceTTest(zx, zy)
end
export RobustTailedTestOfVariance
