using StatsBase
# Will have one function for each baseline method, which takes an input (train) feature matrix and then defines a function/rule for re-basing another input (test) matrix.
function reScale(F::AbstractArray)

end

function reStandardise(F::AbstractArray)
    idxs = vec(nanrows(F) .| constantrows(F))
    𝛔 = vec(StatsBase.std(F, dims=2))
    𝛍 = vec(mean(𝛔, dims=2))
    𝛔[idxs] .= Inf
    return X -> normalise(X, 𝛍, 𝛔, standardise, 2)
end
export reStandardise