using StatsBase
import StatsBase.std
using DimensionalData

function reStandardise(F::AbstractArray)
    idxs = vec(nanrows(F) .| constantrows(F))
    𝛔 = StatsBase.std(F, dims=2)
    𝛍 = mean(F, dims=2)
    𝛔[idxs] .= Inf
    return X -> normalise(X, 𝛍, 𝛔, standardise, 2)
end
export reStandardise

function reZero(F::AbstractArray, α::Float64=10.0)
    # We want to adjust a baseline data set so that features have a variance of 0 if the constant baseline is similar to the high dimensional baseline  (otherwise, the baseline does not change variance).
    # For this, a hyperbolic function of constant and high dimensional baseline variances is used to scale the output
    𝛔 = vec(StatsBase.std(F, dims=2))
    f(X) =  begin
                #C = 1 .- min.((1,), 𝛔./(vec(StatsBase.std(X, dims=2))))
                C = exp.(-α.*𝛔./(vec(StatsBase.std(X, dims=2))))
                𝛍 = vec(mean(X, dims=2))
                return normalise(X, 𝛍, C, standardise, 2)
            end
    return f
end
export reZero


# ------------------------------------------------------------------------------------------------ #
#                            Combine a high and low dimensional baseline                           #
# ------------------------------------------------------------------------------------------------ #
function reScale(x::AbstractVector, f::Function=_self)
    σ = std(x)
    σ′ = f(σ)
    println(σ)
    return σ′.*x./σ
end
function reScale(F::AbstractArray, f::Vector)
    F′ = deepcopy(F)
    for r = 1:size(F, 1)
        F′[r, :] = reScale(F′[r, :] , f[r])
    end
   return F′
end
function reScale(F::DimArray, f::DimArray{T, 1}) where {T}
    (Fᵣ, fᵣ) = intersectFeatures(F, f) # Assumes fᵣ has all of the features of Fᵣ
    reScale(Fᵣ, vec(fᵣ))
end
function hiloScale(Fₗ::Array{Float64, 2}, Fₕ::Array{Float64, 2},
                    interval::Function=(x, y) -> NonstationaryProcesses.rampInterval(0, 1, x, y))
    # interval gives a function of σ, the test variance, with parameters σₗ and σₕ
    if size(Fₗ, 1) != size(Fₕ, 1)
        error("High and low dimensional baselines do not have the same number of features")
    end
    𝛔ₗ, 𝛔ₕ = std(Fₗ, dims=2), std(Fₕ, dims=2)
    𝐟 = interval.(vec(𝛔ₗ), vec(𝛔ₕ))
    return F -> reScale(F, 𝐟)
end
function hiloScale(Fₗ::DimArray{Float64, 2}, Fₕ::DimArray{Float64, 2},
    interval::Function=(x, y) -> NonstationaryProcesses.rampInterval(0, 1, x, y))
    # interval gives a function of σ, the test variance, with parameters σₗ and σₕ
    if any(Catch22.featureDims(Fₗ) .!= Catch22.featureDims(Fₕ))
        error("High and low dimensional baselines do not have the same features")
    end
    𝛔ₗ, 𝛔ₕ = std(Fₗ, dims=2), std(Fₕ, dims=2)
    𝛔ₕ[𝛔ₕ .< 𝛔ₗ] .= Inf
    #𝛔ₕ[𝛔ₕ .< 2.0*𝛔ₗ] .= Inf # Will need a better threshold
    𝐟 = interval.(vec(𝛔ₗ), vec(𝛔ₕ))
    𝐟 = Catch22.featureVector(𝐟, Catch22.featureDims(Fₗ))
    return F -> reScale(F, 𝐟)
end
export hiloScale

# ------------------------------------------------------------------------------------------------ #
#                                    Scale a baseline using PCA                                    #
# ------------------------------------------------------------------------------------------------ #
function orthonormalise(F::AbstractArray, dimensionalityReduction=principalComponents)
    M = dimensionalityReduction(F)
    F̂ = embed(M, F)
    return (F̂, M)
end
function orthonormalise(F::DimArray{Float64, 2}, dimensionalityReduction=principalComponents)
    F̂, M = orthonormalise(Array(F), dimensionalityReduction)
    F̂ = Catch22.featureMatrix(F̂, [Symbol("PC$x") for x ∈ 1:size(F̂, 1)])
    return F̂, M
end
export orthonormalise

function orthonormalBaseline(F::DimArray{Float64, 2}, dimensionalityReduction=principalComponents)
    function 𝑏(F_test)
        F_test, F = intersectFeatures(F_test, F)
        F̂, M = orthonormalise(F, dimensionalityReduction)
        F_test = embed(M, Array(F_test))
        F_out = Catch22.featureMatrix(F_test, [Symbol("PC$x") for x ∈ 1:size(F_test, 1)])
    end
    return 𝑏
end
export orthonormalBaseline

function orthonormalHiloBaseline(F::DimArray, ℱₗ::DimArray, ℱₕ::DimArray; interval::Function=(x, y) -> NonstationaryProcesses.rampInterval(0, 1, x, y))
    F, ℱₗ, ℱₕ = intersectFeatures(F, ℱₗ, ℱₕ) # Intersects to the feature set of F
    ℱₕ′, M = orthonormalise(Array(ℱₕ))
    ℱ′ = Catch22.featureMatrix(ℱₕ′, [Symbol("PC$x") for x ∈ 1:size(ℱₕ′, 1)])
    F′ = embed(M, F)
    ℱ′ₗ = embed(M, ℱₗ)
    𝑏′ = hiloScale(Array(ℱ′ₗ), Array(ℱₕ′), interval)
    return 𝑏′(F′)
end
orthonormalHiloBaseline(ℱₗ::DimArray, ℱₕ::DimArray) = F -> orthonormalHiloBaseline(F, ℱₗ, ℱₕ)
export orthonormalHiloBaseline




# ------------------------------------------------------------------------------------------------ #
#                               Filter features using correlations/MI                              #
# ------------------------------------------------------------------------------------------------ #
function meanDependence(f, F; metric=StatsBase.corspearman)
    ρ = [metric(f, f′) for f′ ∈ eachrow(F)]
    ρ̄ = mean(abs.(ρ))
end

function meanDependence(F; metric=StatsBase.corspearman)
    𝐟 = zeros(size(F, 1))
    𝐟 = [meanDependence(F[x, :], F[setdiff(1:size(F, 1), [x]), :]; metric) for x ∈ 1:size(F, 1)]
end
export meanDependence

function dependencyFilter(F, threshold=0.3; metric=StatsBase.corspearman, iterations=Inf, direction=:min)
    # Iteratively filter features according to algorithm in Ben's thesis, page 214
    iteration = 0
    if direction == :min
        direction = 1
    elseif direction == :max
        direction = -1
    end
    Ī = zeros(size(F, 1)) .+ (-direction + 1)/2
    while any(direction.*Ī .< direction.*threshold) && iteration < iterations # Yes Ī keeps changing size
        Ī = meanDependence(F; metric)
        I, f = findmin(direction.*Ī)
        F = F[setdiff(1:lastindex(F, 1), [f]), :]
        iteration += 1
    end
    return F
end
export dependencyFilter
