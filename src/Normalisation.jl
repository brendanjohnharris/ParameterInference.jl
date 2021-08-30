using Statistics
using StatsBase
using Tullio
# ------------------------------------------------------------------------------------------------ #
#                             Functions for normalising feature vectors                            #
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------- General normalisation ------------------------------------ #
tanh(x::Number, centre::Number=0, scale::Number=1) = Base.tanh(x - centre)*scale; export tanh
standardise(x::Number, centre::Number=0, scale::Number=1) = (x - centre)/(scale); export standardise
logistic(x::Number, centre::Number=0, scale::Number=1) = 1/(1 + exp(-(x-centre)/scale)); export logistic
centre(x::Number, centre::Number=0, scale::Number=1) = x - centre; export centre

function normalise(x::Union{AbstractVector, Number}, f::Function=standardise, μ::Number=mean(x), σ::Number=std(x))
    y = f.(x, μ, σ)
end
function normalise(X::AbstractArray, f::Function=standardise, dim::Int=2, args...) # Rows by default
    Y = mapslices(x -> normalise(x, f, args...), X, dims=dim)
end
function normalise(X::AbstractArray, 𝛍::AbstractArray, 𝛔::AbstractArray, f::Function=standardise, dim::Int=2)
    # Need a generalised mapslices
    if dim == 1
        @tullio Y[i, j] := normalise(X[i, j], f, 𝛍[j], 𝛔[j])
    elseif dim == 2
        @tullio Y[i, j] := normalise(X[i, j], f, 𝛍[i], 𝛔[i])
    end
end
function normalise(X::AbstractFeatureArray, 𝛍::AbstractFeatureArray, 𝛔::AbstractFeatureArray, f::Function=standardise, dim::Int=2)
    X, 𝛍 = intersectFeatures(X, 𝛍)
    X, 𝛔 = intersectFeatures(X, 𝛔)
    X, 𝛍 = intersectFeatures(X, 𝛍) # In case sigma is different from mu in features
    normalise(X, vec(𝛍), vec(𝛔), f, dim)
end
export normalise

unitInterval(x::AbstractVector) = normalise(x, standardise, min(x...), abs(-(extrema(x)...)))
unitInterval(X::AbstractMatrix, dim::Int=2) = mapslices(unitInterval, X, dims=dim)
export unitInterval

standardise(x::AbstractVector, args...) = normalise(x, standardise, args...)
standardise(X::AbstractMatrix, dim::Int=2) = mapslices(standardise, X, dims=dim)
export standardise

sigmoidNormalise(x::AbstractVector, args...) = normalise(x, logistic, args...)
sigmoidNormalise(X::AbstractMatrix, dim::Int=2) = mapslices(sigmoidNormalise, X, dims=dim)
export sigmoidNormalise





function robustNormalise(x::Union{AbstractVector, Number}, f::Function=standardise, μ::Number=median(x), σ::Number=iqr(x))
    y = f.(x, μ, σ/1.35)
end
function robustNormalise(X::AbstractArray, f::Function=standardise, dim::Int=2, args...) # Rows by default
    Y = mapslices(x -> robustNormalise(x, f, args...), X, dims=dim)
end
function robustNormalise(X::AbstractArray, 𝛍::AbstractArray, 𝛔::AbstractArray, f::Function=standardise, dim::Int=2)
    if dim == 1
        @tullio Y[i, j] := robustNormalise(X[i, j], f, 𝛍[j], 𝛔[j])
    elseif dim == 2
        @tullio Y[i, j] := robustNormalise(X[i, j], f, 𝛍[i], 𝛔[i])
    end
end
# function robustNormalise(X::AbstractFeatureArray, 𝛍::AbstractFeatureArray, 𝛔::AbstractFeatureArray, f::Function=standardise, dim::Int=2)
#     X, 𝛍 = intersectFeatures(X, 𝛍)
#     X, 𝛔 = intersectFeatures(X, 𝛔)
#     X, 𝛍 = intersectFeatures(X, 𝛍) # In case sigma is different from mu in features
#     robustNormalise(X, vec(𝛍), vec(𝛔), f, dim)
# end
export robustNormalise

robustStandardise(x::AbstractVector, args...) = robustNormalise(x, standardise, args...)
robustStandardise(X::AbstractMatrix, dim::Int=2) = mapslices(standardise, X, dims=dim)
export robustStandardise

robustSigmoidNormalise(x::AbstractVector, args...) = robustNormalise(x, logistic, args...)
robustSigmoidNormalise(X::AbstractMatrix, dim::Int=2) = mapslices(robustSigmoidNormalise, X, dims=dim)
export robustSigmoidNormalise

# ------------------------------------ Scale to L1 unit vector ----------------------------------- #
unitL1(x::AbstractVector) = x./sum(x)
unitL1(X::AbstractMatrix) = mapslices(unitL1, X, dims=2)
export unitL1


# ------------------------------------ Scale to unit interval ------------------------------------ #
# function unitInterval(x::AbstractVector)
#     y = x .- min(x...)
#     y = y./max(y...)
# end
# unitInterval(X::AbstractMatrix) = mapslices(unitInterval, X, dims=2)
# export unitInterval






# ------------------------------------------------------------------------------------------------ #
#                                  Functions for filtering arrays                                  #
# ------------------------------------------------------------------------------------------------ #
function constantrows(F::AbstractArray; tol=1e-10)
    idxs = Array(StatsBase.std(F, dims=2) .< tol)
end
export constantrows

function _noconstantrows(F::AbstractArray; tol=1e-10)
    idxs = constantrows(F; tol=tol)
    if any(idxs)
        @warn "$(+(idxs...)) constant rows are being removed"
        return F -> F[collect(.!idxs)[:], :]
    end
    return F -> F
end
noconstantrows(F::AbstractArray; tol=1e-10) = _noconstantrows(F; tol)
export noconstantrows


function nanrows(F::AbstractArray)
    idxs = any(isnan.(Array(F)), dims=2)
end
export nanrows

function infrows(F::AbstractArray)
    idxs = any(isinf.(Array(F)), dims=2)
end
export nanrows

function _nonanrows(F::AbstractArray)
    idxs = nanrows(F)
    if any(idxs)
        @warn "$(+(idxs...)) NaN rows will be removed" # Maybe more detail?
        return F -> F[collect(.!idxs)[:], :]
    end
    return F -> F
end
nonanrows(F::AbstractArray) = _nonanrows(F)(F)
export nonanrows

function _noinfrows(F::AbstractArray)
    idxs = infrows(F)
    if any(idxs)
        @warn "$(+(idxs...)) Inf rows will be removed"
        return F -> F[collect(.!idxs)[:], :]
    end
    return F -> F
end
noinfrows(F::AbstractArray) = _noinfrows(F)(F)
export noinfrows

function nonans(F::AbstractVector)
    idxs = isnan.(Array(F))
    if any(idxs)
        @warn "$(+(idxs...)) NaN elements are being removed"
    end
    return F[.!idxs]
end
export nonans

function nomissing(F::AbstractVector)
    idxs = ismissing.(Array(F))
    # if any(idxs)
    #     @warn "$(+(idxs...)) missing elements are being removed"
    # end
    return F[.!idxs]
end
export nomissing

function noinf(F::AbstractVector)
    idxs = isinf.(Array(F))
    return F[.!idxs]
end
export noinf
