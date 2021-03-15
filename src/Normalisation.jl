using Statistics
# ------------------------------------------------------------------------------------------------ #
#                             Functions for normalising feature vectors                            #
# ------------------------------------------------------------------------------------------------ #

zscore(x::AbstractVector{Float64}, μ::Float64=mean(x), σ::Float64=std(x)) = (x .- μ)./(σ)
zscore(X::AbstractArray{Float64, 2}) = mapslices(zscore, X, dims=2) # Normalise rows i.e. feature vectors
# function zscore!(X::AbstractArray{Float64, 2})
#     X = zscore(X)
# end
export zscore, zscore!



# ------------------------------------- Sigmoid Normalisation ------------------------------------ #
#..............



# ------------------------------------ Scale to L1 unit vector ----------------------------------- #
unitL1(x::AbstractVector{Float64}) = x./sum(x)
unitL1(X::AbstractArray{Float64, 2}) = mapslices(unitL1, X, dims=2)
export unitL1


# ------------------------------------ Scale to unit interval ------------------------------------ #
function unitInterval(x::AbstractVector{Float64})
    y = x .- min(x...)
    y = y./max(y...)
end
unitInterval(X::AbstractArray{Float64, 2}) = mapslices(unitInterval, X, dims=2)
export unitInterval






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
