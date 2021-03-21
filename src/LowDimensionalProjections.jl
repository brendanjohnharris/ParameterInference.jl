using MultivariateStats
using Distances
using Catch22
using ManifoldLearning


# Function wrapping projectors
function project(F::AbstractArray, projectionFunc, args...; kwargs...)
    M = projectionFunc(F, args...; kwargs...)
end
export project


# ------------------------------------------------------------------------------------------------ #
#                                                 .                                                #
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
#                                                PCA                                               #
# ------------------------------------------------------------------------------------------------ #
function principalComponents(F::AbstractArray; pratio=1.0, kwargs...)
    M = MultivariateStats.fit(MultivariateStats.PCA, F; pratio=pratio, kwargs...)
end
export principalComponents

function embed(M::MultivariateStats.PCA{Float64}, F::AbstractArray, PCs::Union{Int, Vector{Int64}, UnitRange}=1:length(M.prinvars))
    P = MultivariateStats.projection(M)
    P = P[:, PCs]
    D = (P'*(F .- mapslices(mean, F, dims=2)))
    if size(D, 1) == 1
        D = D[:]
    end
    return D
end
export embed


# ------------------------------------------------------------------------------------------------ #
#                                        Explained Variance                                        #
# ------------------------------------------------------------------------------------------------ #
function explainedVariance(M::MultivariateStats.PCA{Float64})
    ev = principalvars(M)
    ev = cumsum(unitL1(ev))
end
export explainedVariance
# add method to explainedVariance for new data


# ------------------------------------------------------------------------------------------------ #
#                                          Feature Weights                                         #
# ------------------------------------------------------------------------------------------------ #
function PCfeatureWeights(I::Inference, pc::Int=1)
    P = projection(I.model)
    Catch22.featureVector(P[:, pc], val(dims(I.F̂, :feature)))
end
export PCfeatureWeights



# ------------------------------------------------------------------------------------------------ #
#                                                 .                                                #
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
#                                              Isomap                                              #
# ------------------------------------------------------------------------------------------------ #
function isomap(F::AbstractArray; maxoutdim=size(F, 1)÷2, kwargs...)
    # Strange bug here; cant have the maxoutdim too high otherwise:
    #ERROR: DomainError with -8.109628524266554e-10:
    #sqrt will only return a complex result if called with a complex argument. Try sqrt(Complex(x)).
    # Watch out for this in the future
    M = MultivariateStats.fit(ManifoldLearning.Isomap, Array(F); maxoutdim=maxoutdim, kwargs...)
end
export isomap

function embed(M::ManifoldLearning.Isomap, F::AbstractArray, PCs::Union{Int, Vector{Int64}, UnitRange}=1:ManifoldLearning.outdim(M))
    D = ManifoldLearning.transform(M, F)
    D = D[PCs, :]
    if size(D, 1) == 1
        D = D[:]
    end
    return D
end
# If you just want the original data transformed, this is slightly faster:
function embed(M::ManifoldLearning.Isomap, PCs::Union{Int, Vector{Int64}, UnitRange}=1:ManifoldLearning.outdim(M))
    D = ManifoldLearning.transform(M)
    D = D[PCs, :]
    if size(D, 1) == 1
        D = D[:]
    end
    return D
end
export embed






function residualVariance(M, F::AbstractArray, nPCs::Int64)
    D = embed(M, Array(F), 1:nPCs) # Model self-embedding can be run more quickly...
    if typeof(D) <: Vector
        D = D'
    end
    dD = pairwise(Euclidean(), D, D, dims=2)
    dF = pairwise(Euclidean(), F, F, dims=2)
    residualVariance(dF, dD)
end
function residualVariance(M, F::AbstractArray)
    𝛔² = residualVariance.((M,), (F,), 1:outdim(M))
end
function residualVariance(dF::Array, dD::Array{Float64, 2})
    # Calculate a distance matrix for F and D, and correlate them
    σ² = 1 - (cor(dF[:], dD[:]))^2
    # Sqrt of 1 minus the correlation between distances in feature space and the low dimensional space, squared
end
export residualVariance