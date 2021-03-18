using Statistics
using StatsBase
# ------------------------------------------------------------------------------------------------ #
#                             Functions for normalising feature vectors                            #
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------- General normalisation ------------------------------------ #
tanh(x::Number, centre::Number=0, scale::Number=1) = Base.tanh(x - centre)*scale; export tanh
standardise(x::Number, centre::Number=0, scale::Number=1) = (x - centre)/(scale); export standardise
logistic(x::Number, centre::Number=0, scale::Number=1) = 1/(1 + exp(-(x-centre)/scale)); export logistic

function normalise(x::AbstractVector, f::Function=standardise, μ::Number=mean(x), σ::Number=std(x))
    y = f.(x, μ, σ)
end
function normalise(X::AbstractArray, f::Function=standardise, dim::Int=2, args...) # Rows by default
    Y = mapslices(x -> normalise(x, f, args...), X, dims=dim)
end
export normalise

unitInterval(x::AbstractVector{Float64}) = normalise(x, standardise, min(x...), abs(-(extrema(x)...)))
unitInterval(X::AbstractArray{Float64, 2}, dim::Int=2) = mapslices(standardise, X, dims=dim)
export unitInterval

standardise(x::AbstractVector{Float64}, args...) = normalise(x, standardise, args...)
standardise(X::AbstractArray{Float64, 2}, dim::Int=2) = mapslices(standardise, X, dims=dim)
export standardise

sigmoidNormalise(x::AbstractVector{Float64}, args...) = normalise(x, logistic, args...)
sigmoidNormalise(X::AbstractArray{Float64, 2}, dim::Int=2) = mapslices(sigmoidNormalise, X, dims=dim)
export sigmoidNormalise





function robustNormalise(x::AbstractVector, f::Function=standardise, μ::Number=median(x), σ::Number=iqr(x))
    y = f.(x, μ, σ/1.35)
end
function robustNormalise(X::AbstractArray, f::Function=standardise, dim::Int=2, args...) # Rows by default
    Y = mapslices(x -> robustNormalise(x, f, args...), X, dims=dim)
end
export robustNormalise

robustStandardise(x::AbstractVector{Float64}, args...) = robustNormalise(x, standardise, args...)
robustStandardise(X::AbstractArray{Float64, 2}, dim::Int=2) = mapslices(standardise, X, dims=dim)
export robustStandardise

robustSigmoidNormalise(x::AbstractVector{Float64}, args...) = robustNormalise(x, logistic, args...)
robustSigmoidNormalise(X::AbstractArray{Float64, 2}, dim::Int=2) = mapslices(robustSigmoidNormalise, X, dims=dim)
export robustSigmoidNormalise

# ------------------------------------ Scale to L1 unit vector ----------------------------------- #
unitL1(x::AbstractVector{Float64}) = x./sum(x)
unitL1(X::AbstractArray{Float64, 2}) = mapslices(unitL1, X, dims=2)
export unitL1


# ------------------------------------ Scale to unit interval ------------------------------------ #
# function unitInterval(x::AbstractVector{Float64})
#     y = x .- min(x...)
#     y = y./max(y...)
# end
# unitInterval(X::AbstractArray{Float64, 2}) = mapslices(unitInterval, X, dims=2)
# export unitInterval






# ------------------------------------------------------------------------------------------------ #
#                                  Functions for filtering arrays                                  #
# ------------------------------------------------------------------------------------------------ #

function nonanrows(F::AbstractArray)
    idxs = any(isnan.(Array(F)), dims=2)
    if any(idxs)
        @warn "$(+(idxs...)) NaN rows are being removed" # Maybe more detail?
        return F[collect(.!idxs)[:], :]
    end
    return F
end
export nonanrows

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
