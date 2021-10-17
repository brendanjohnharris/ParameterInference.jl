using MultivariateStats
using Distances
using Catch22
using ManifoldLearning
using LinearAlgebra
using LowRankModels
using SparseArrays


# Function wrapping projectors
function project(F::AbstractArray, projectionFunc, args...; kwargs...)
    M = projectionFunc(F, args...; kwargs...)
end
function project(F::AbstractFeatureArray, projectionFunc, args...; kwargs...)
    M = projectionFunc(Array(F), args...; kwargs...)
end
export project


"""
PCA
"""
function principalcomponents(F::AbstractArray; pratio=1.0, kwargs...)
    M = MultivariateStats.fit(MultivariateStats.PCA, F; pratio=pratio, kwargs...)
end
principalcomponents(F::AbstractFeatureArray, args...; kwargs...) = principalcomponents(Array(F), args...; kwargs...)
export principalcomponents

function embed(M::MultivariateStats.PCA, F::AbstractArray, PCs::Union{Int, Vector{Int64}, UnitRange}=1:length(M.prinvars))
    P = MultivariateStats.projection(M)
    P = P[:, PCs]
    D = (P'*(F .- mean(M)))
    if size(D, 1) == 1
        D = D[:]
    end
    return D
end

export PCA

function embed(M::MultivariateStats.PCA, F::AbstractFeatureArray, PCs::Union{Int, Vector{Int64}, UnitRange}=1:length(M.prinvars); kwargs...)
    D = embed(M, Array(F), PCs; kwargs...)
    Catch22.featureMatrix(D, [Symbol("PC$x") for x ∈ 1:size(D, 1)])
end
export embed


"""
Explained Variance
"""
function explainedVariance(M::MultivariateStats.PCA)
    ev = principalvars(M)
    ev = cumsum(unitL1(ev))
end
export explainedVariance

# add method to explainedVariance for new data

function explainedVariance(F::AbstractArray)
# The varainces of each feature as a proportion of the total variance
    Σ² = StatsBase.cov(F, dims=2)
    ev = diag(Σ²)./sum(diag(Σ²))
end

export principalvars

"""
Weights on features
"""
function featureweights(model::MultivariateStats.PCA, pc=1:outdim(model))
    P = projection(model)[:, pc]
end
function featureweights(I::Inference, pc::Int=1)
    P = projection(I.model)
    Catch22.featureVector(P[:, pc], val(dims(I.F̂, :feature)))
end
export featureweights



"""
Isomap
"""
function isomap(F::AbstractArray; maxoutdim=max(size(F, 1)÷2, 10), kwargs...)
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
export Isomap
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
    if D isa Vector
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











function robustprincipalcomponents(F::AbstractArray, npcs=size(F, 1)::Int, scale::Float64=1.0; abs_tol=1e-4^3, rel_tol=1e-4^3, max_iter=10000, kwargs...)
	loss = HuberLoss()
	r = ZeroReg()
    # Start with PCA
    m = GLRM(F, QuadLoss(), ZeroReg(), ZeroReg(), npcs; kwargs...)
    fit!(m)
    # Then do the other thing
    m = GLRM(A, loss, r, r, k; kwargs...)
    fit!(m, ProxGradParams(;abs_tol, rel_tol, max_iter))
	return m
end
MultivariateStats.projection(M::LowRankModels.GLRM) = inv(M.X)
function embed(M::LowRankModels.GLRM, F::AbstractArray, PCs::Union{Int, Vector{Int64}, UnitRange}=1:M.k)
    @assert M.k == size(F,  1) # For now, to invert X it must be square whcih means keeping all PCs
    P = projection(M)
    P = P[:, PCs]
    D = P'*F
    if size(D, 1) == 1
        D = D[:]
    end
    return D
end
export robustprincipalcomponents


nanmedian = x -> all(isnan.(x)) ? NaN : median(x[.!isnan.(x)])
naniqr = x -> all(isnan.(x)) ? NaN : iqr(x[.!isnan.(x)])

function nanprincipalcomponents(F::AbstractArray, args...; kwargs...)
    # Convert F to a sparse array, sending NaNs to sparse elements
    init = deepcopy(F) |> Array # Chunky?
    for i ∈ 1:size(init, 1)
        repnans = all(isnan.(init[i, :])) ? randn(sum(isnan.(init[i, :]))).*eps() : nanmedian(init[i, :][.!isnan.(init[i, :])]) # If all nans, just use some tiny noise.
        init[i, isnan.(init[i, :])] .= repnans
    end
    A = SparseMatrixCSC(F)
    SparseArrays.fkeep!(A, (i,j,x) -> !isnan(x))
    m = _nanprincipalcomponents(A, init, args...; kwargs...)
end

function _nanprincipalcomponents(A::AbstractArray, init, npcs=size(A, 1)::Int, scale::Float64=1.0; abs_tol=1e-4^3, rel_tol=1e-4^3, max_iter=20000, kwargs...)
    m = GLRM(A, QuadLoss(), ZeroReg(), ZeroReg(), npcs; kwargs...)
    U, D, V = svd(init)
    D = D |> Diagonal
    m.X, m.Y = collect((U*sqrt(D))'), sqrt(D)*V';
    fit!(m, ProxGradParams(;abs_tol, rel_tol, max_iter))
    return m
end
export nanprincipalcomponents

function outlierprincipalcomponents(F::AbstractArray, args...; kwargs...)
    # * The goal is the remove any outliers (say, greater than 10 iqrs from the median) and then do PCA
    A = deepcopy(F)
    centres = mapslices(nanmedian, F, dims=2)
    scales = mapslices(naniqr, F, dims=2)
    for i ∈ 1:size(A, 1)
        A[i, abs.(A[i, :] .- centres[i]) .>= 5*scales[i]] .= NaN
    end
    return nanprincipalcomponents(A, args...; kwargs...)
end
export outlierprincipalcomponents


"""
Only keep the PC's explaining more than `thresh` proportion of variance.
"""
function significantprincipalcomponents(F::AbstractArray; thresh=1/size(F, 1), kwargs...)
    M = principalcomponents(F; kwargs...)
    v = principalvars(M)./sum(principalvars(M))
    npcs = sum(v .> thresh)
    M = principalcomponents(F; maxoutdim=npcs, kwargs...)
end
significantprincipalcomponents(F::AbstractFeatureArray, args...; kwargs...) = significantprincipalcomponents(Array(F), args...; kwargs...)
export significantprincipalcomponents
