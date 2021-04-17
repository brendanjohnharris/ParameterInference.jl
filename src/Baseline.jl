using StatsBase
import StatsBase.std

function reStandardise(F::AbstractArray)
    idxs = vec(nanrows(F) .| constantrows(F))
    𝛔 = vec(StatsBase.std(F, dims=2))
    𝛍 = vec(mean(F, dims=2))
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
    σ = std(F)
    σ′ = f(σ)
    return σ′.*x
end
function reScale(F::AbstractArray, f::Vector{Function})
    F′ = copy(F)
    for r = 1:size(F, 1)
        F′[r, :] = reScale(F′[r, :] , f[r])
    end
    return F′
end
function hiloScale(Fₗ::AbstractArray{Float64, 2}, Fₕ::AbstractArray{Float64, 2},
                    interval::Function=(x, y) -> NonstationaryProcesses.intervalRamp(0, 1, x, y))
    # interval gives a function of σ, the test variance, with parameters σₗ and σₕ
    if size(Fₗ, 1) != size(Fₕ, 1)
        error("High and low dimensional baselines do not have the same number of features")
    endrampInter
    𝛔ₗ, 𝛔ₕ = std(Fₗ, dims=2), std(Fₕ, dims=2)
    𝐟 = interval.(𝛔ₗ, 𝛔ₕ)
    return reScale.(F, 𝐟)
end
