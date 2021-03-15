using MultivariateStats
using Distances


# Function wrapping projectors
function project(F::AbstractArray, projectionFunc, args...; kwargs...)
    M = projectionFunc(F, args...; kwargs...)
end
export project

# ------------------------------------------------------------------------------------------------ #
#                                            Projectors                                            #
# ------------------------------------------------------------------------------------------------ #
function principalComponents(F::AbstractArray; pratio=1.0, kwargs...)
    M = MultivariateStats.fit(MultivariateStats.PCA, F; pratio=pratio, kwargs...)
end
export principalComponents
function embed(M, F::AbstractArray, PCs::Union{Vector{Int64}, UnitRange}=1:length(M.prinvars))
    P = MultivariateStats.projection(M)
    P = P[:, PCs]
    D = (P'*(F .- mapslices(mean, F, dims=2)))
    if size(D, 1) == 1
        D = D[:]
    end
end
export embed


# ------------------------------------------------------------------------------------------------ #
#                                        Explained Variance                                        #
# ------------------------------------------------------------------------------------------------ #
function explainedVariance(M)
    ev = principalvars(M)
    ev = cumsum(unitL1(ev))
end
export explainedVariance
# add method to explainedVariance for new data



# ------------------------------------------------------------------------------------------------ #
#                                         Residual Variance                                        #
# ------------------------------------------------------------------------------------------------ #
function residualVariance(dF::Array, dD::Array{Float64, 2})
    # Calculate a distance matrix for F and D, and correlate them
    σ² = 1 - (cor(dF[:], dD[:]))^2
    # Sqrt of 1 minus the correlation between distances in feature space and the low dimensional space, squared
end

function residualVariance(M, F::AbstractArray, nPCs::Int64)
    D = embed(M, F, 1:nPCs)
    dD = pairwise(Euclidean(), D, D, dims=2)
    dF = pairwise(Euclidean(), F, F, dims=2)
    residualVariance(dF, dD)
end
function residualVariance(M, F::AbstractArray)
    𝛔² = residualVariance.((M,), (F,), 1:outdim(M))
end
export residualVariance

