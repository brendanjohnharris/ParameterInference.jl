using StatsBase
using Catch22
import StatsBase.std
using LinearAlgebra

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
function reScale(x::AbstractVector, f::Function=identity)
    σ = std(x)
    σ′ = f(x)
    if σ′ == σ == 0.0
        σ′ = σ = 1.0 # Catch the limit
    end
    return σ′.*x./σ
end
function reScale(F::AbstractArray, f::Vector)
    F′ = deepcopy(F)
    for r = 1:size(F, 1)
        F′[r, :] = reScale(F′[r, :] , f[r])
    end
   return F′
end
function reScale(F::AbstractFeatureArray, f::AbstractFeatureVector)
    (Fᵣ, fᵣ) = intersectFeatures(F, f) # Assumes f has all of the features of F
    reScale(Fᵣ, vec(fᵣ))
end

function NonstationaryProcesses.rampInterval(Fₗ, Fₕ, F)
    𝛔ₗ, 𝛔ₕ = std(Array(Fₗ), dims=2), std(Array(Fₕ), dims=2)
    𝛔ₕ[𝛔ₕ .< 𝛔ₗ] .= Inf
    𝐟 = [𝐟 -> rampInterval(0.0, 1.0, 𝛔ₗ[i], 𝛔ₕ[i])(std(𝐟)) for i ∈ 1:size(F, 1)]
end

function intervalscale(Fₗ::Array{Float64, 2}, Fₕ::Array{Float64, 2},
                    interval::Function=rampInterval)
    if size(Fₗ, 1) != size(Fₕ, 1)
        error("High and low dimensional baselines do not have the same number of features")
    end
    return F -> reScale(F, interval(Fₗ, Fₕ, F))
end
function intervalscale(Fₗ::AbstractFeatureMatrix, Fₕ::AbstractFeatureMatrix,
    interval::Function=rampInterval)
    if any(Catch22.featureDims(Fₗ) .!= Catch22.featureDims(Fₕ))
        error("High and low dimensional baselines do not have the same features")
    end
    return F -> reScale(F,  Catch22.featureVector(interval(Fₗ, Fₕ, F), Catch22.featureDims(F)))
end
export intervalscale

# ------------------------------------------------------------------------------------------------ #
#                                    Scale a baseline using PCA                                    #
# ------------------------------------------------------------------------------------------------ #
function orthogonalise(F::AbstractArray, dimensionalityReduction=principalcomponents)
    M = dimensionalityReduction(F)
    F̂ = embed(M, F)
    return (F̂, M)
end
function orthogonalise(F::AbstractFeatureMatrix, dimensionalityReduction=principalcomponents)
    F̂, M = orthogonalise(Array(F), dimensionalityReduction)
    F̂ = Catch22.featureMatrix(F̂, [Symbol("PC$x") for x ∈ 1:size(F̂, 1)])
    return F̂, M
end
export orthogonalise

function orthogonalBaseline(F::AbstractArray, dimensionalityReduction=principalcomponents)
    function 𝑏(F_test)
        @assert size(F, 1) == size(F_test, 1)
        F̂, M = orthogonalise(F, dimensionalityReduction)
        F_out = embed(M, Array(F_test))
    end
    return 𝑏
end
function orthogonalBaseline(F::AbstractFeatureMatrix, dimensionalityReduction=principalcomponents)
    function 𝑏(F_test)
        F̂_test, F̂ = intersectFeatures(F_test, F)
        F̂, M = orthogonalise(F̂, dimensionalityReduction)
        F_out = embed(M, Array(F̂_test))
        F_out = Catch22.featureMatrix(F_out, [Symbol("PC$x") for x ∈ 1:size(F_out, 1)])
    end
    return 𝑏
end
export orthogonalBaseline

function orthonormalHiloBaseline(F::AbstractFeatureArray, ℱₗ::AbstractFeatureArray, ℱₕ::AbstractFeatureArray; interval::Function=(x, y) -> NonstationaryProcesses.rampInterval(0, 1, x, y))
    F, ℱₗ, ℱₕ = intersectFeatures(F, ℱₗ, ℱₕ) # Intersects to the feature set of F
    ℱₕ′, M = orthogonalise(Array(ℱₕ))
    ℱ′ = Catch22.featureMatrix(ℱₕ′, [Symbol("PC$x") for x ∈ 1:size(ℱₕ′, 1)])
    F′ = embed(M, F)
    ℱ′ₗ = embed(M, ℱₗ)
    𝑏′ = intervalscale(Array(ℱ′ₗ), Array(ℱₕ′), interval)
    return 𝑏′(F′)
end
orthonormalHiloBaseline(ℱₗ::AbstractFeatureArray, ℱₕ::AbstractFeatureArray; kwargs...) = F -> orthonormalHiloBaseline(F, ℱₗ, ℱₕ; kwargs...)
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





# * We have four baselines associated with rescaling variances, plus the PCA whitening rotation that can be performed before all four rescalings

# ------------------------------------------- Rescaling ------------------------------------------ #
"""
Standardise the test features. Alternatively, just use the normalisation field of an Inference.
"""
standardbaseline(F::AbstractArray) = standardise(F, 2)
standardbaseline() = F -> standardbaseline(F)
export standardbaseline


"""
Scale the features so that the variance of a constant baseline is zero. Do this by setting mapping variances less than σₗ to 0, but keeping a gradient of 1.0 afterwards. Zscore features, zscore baseline from features and then map
"""
function lowbaseline(Fₗ::AbstractArray)
    interval = x -> NonstationaryProcesses.rampOn(0.0, 1.0, x, x+1.0)
    function lowscale(F::AbstractArray)
        F̂, F̂ₗ = intersectFeatures(F, Fₗ)
        𝛔 = StatsBase.std(F̂, dims=2)
        𝛍 = StatsBase.mean(F̂, dims=2)
        F̂ₗ = normalise(F̂ₗ, 𝛍, 𝛔, standardise, 2)
        F̂ = normalise(F̂, 𝛍, 𝛔, standardise, 2)
        𝛔 = StatsBase.std(F̂, dims=2)
        𝛍 = StatsBase.mean(F̂, dims=2)
        𝛔ₗ = StatsBase.std(F̂ₗ, dims=2)
        𝛍ₗ = StatsBase.mean(F̂ₗ, dims=2)
        𝐟 = [x -> interval(σ)(StatsBase.std(x)) for σ ∈ 𝛔ₗ][:]
        if F isa AbstractFeatureArray
            𝐟 = Catch22.FeatureVector(𝐟, Catch22.featureDims(F̂ₗ))
        end
        return reScale(F̂, 𝐟)
    end
    return lowscale
end
export lowbaseline

"""
Scale the features so that the variance of a high dimensional baselines is unity. Do this as an inverval with a constant variance of 0.0
"""
function highbaseline(Fₕ::AbstractArray)
    Fₗ = zeros(size(Fₕ))
    intervalscale(Fₗ, Fₕ)
end
function highbaseline(Fₕ::AbstractFeatureArray)
    Fₕ, Fₗ = intersectFeatures(Fₕ, zeros(size(Fₕ)))
    intervalscale(Fₗ, Fₕ)
end
export highbaseline

"""
Scale the features so that their variances map to a rampInterval between the low (0) and high dim (1) baselines.
"""
function intervalbaseline(Fₗ::AbstractArray, Fₕ::AbstractArray)
    intervalscale(Fₗ, Fₕ)
end
export intervalbaseline

# # ---------------------------------- High dim orthonormalisation --------------------------------- #
# """
# Add this to any baseline variables and the Inference normalisation to transform into the high dim. whitened space
# e.g. infer(S, var; parameters, features, baseline=intervalbaseline(𝑜(Fₗ), 𝑜(Fₕ)), normalisation=𝑜) # Note normalisation occurs before baseline
# """
# orthonormaliseto(Fₕ::AbstractArray, dimensionalityReduction=principalcomponents) = orthogonalBaseline(Fₕ, dimensionalityReduction)
# export orthonormaliseto


# ---------------------------------- High dim orthogonalisation --------------------------------- #
"""
Add this to a baseline after constructing a scaling baseline and using that as normalisation
e.g. 𝑏 = intervalbaseline(Fₗ, Fₕ)
infer(S, var; parameters, features, baseline=orthogonaliseto(𝑏(Fₕ)), normalisation=𝑏) # Note normalisation occurs before baseline
"""
orthogonaliseto(Fₕ::AbstractArray, dimensionalityReduction=principalcomponents) = orthogonalBaseline(Fₕ, dimensionalityReduction)
export orthogonaliseto



"""
Interval scaling informed by distributions. The idea is to decrease the scale of a feature if we are unsure of its location between the high and zero dim distributions
"""
function significanceinterval(Fₗ, Fₕ, F)
    𝐟 = rampInterval(Fₗ, Fₕ, F)
    𝑝₀ = [f -> pvalue(VarianceFTest(Fₗ[i, :], f), tail=:right) for i ∈ 1:size(Fₗ, 1)]
    𝑝ₕ = [f -> pvalue(VarianceFTest(Fₕ[i, :], f), tail=:left) for i ∈ 1:size(Fₕ, 1)]
    function out(f, i)
        𝐟[i](f[:]).*𝑝₀[i](f[:]).*𝑝ₕ[i](f[:])
    end
    return [f -> out(f, i) for i ∈ 1:size(𝐟, 1)]
end
export significanceinterval


"""
errorintervalscaling
"""
function errorintervalscaling(Fₗ, Fₕ, F)
    𝐟 = rampInterval(Fₗ, Fₕ, F)
    function out(f, i)
        Δ = rampInterval(0.0, 1.0, 0.0, 1.0)(bootstrapSEσ(f[:])/(std(Fₕ[i, :]) - std(Fₗ[i, :]))) # So it saturates at extremes
        𝐟[i](f[:])*(1 - Δ)
    end
    return [f -> out(f, i) for i ∈ 1:size(𝐟, 1)]
end
export errorintervalscaling


"""
Scale the features so that their variances map to a rampInterval between the low (0) and high dim (1) baselines.
"""
function errorintervalbaseline(Fₗ::AbstractArray, Fₕ::AbstractArray)
    intervalscale(Fₗ, Fₕ, errorintervalscaling)
end
export errorintervalbaseline




"""
What's this?
"""
# Total covariance correction
function dependencyscalingnorotation(𝑏, Fₕ)
    Fₕ′ = 𝑏(Fₕ)
    function g(F)
        F′, Fₕ′ = intersectFeatures(noconstantrows(F), Fₕ′)
        # Fₕ′, F′ = intersectFeatures(noconstantrows(Fₕ′), F′)
        Σₕ² = StatsBase.cov(Array(Fₕ′), dims=2)
        𝑜 = orthogonaliseto(Fₕ′, principalcomponents)
        𝐧 = sum(abs.(Array(Σₕ²)), dims=2)
        𝐧[𝐧 .== 0] .= Inf
        N⁻¹ = FeatureMatrix(inv(sqrt(Diagonal(𝐧[:]))), getnames(Fₕ′))
        return FeatureMatrix(N⁻¹*𝑏(F), getnames(F′))
    end
    return g
end
export dependencyscalingnorotation
