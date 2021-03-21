using Clustering
using Distances

function clusterDistances(D::AbstractMatrix; linkageMetric::Symbol=:average, branchOrder::Symbol=:optimal)
# Like hctsa, should only use optimalleaforder in N < 2000
    if size(D, 2) > 2000
        @warn "Consider using the :r branchOrder with more than 2000 columns"
    end
    h = Clustering.hclust(D; linkage=linkageMetric, branchorder=branchOrder)
    order = h.order
end

function clusterPairwise(X::AbstractMatrix, distanceMetric=Distances.CorrDist(); dim=2, kwargs...)
    D = pairwise(distanceMetric, X, dims=dim)
    clusterDistances(D; kwargs...)
end

function clusterReorder(X::AbstractMatrix, distanceMetric=Distances.CorrDist(); dim=1, kwargs...)
    idxs = clusterPairwise(X, distanceMetric; dim=dim, kwargs...)
    if dim==2
        X[:, idxs] # Orders columns by default
    elseif dim==1
        X[idxs, :]
    end
end

function clusterReorder(X::AbstractMatrix, D::AbstractMatrix; dim=2, kwargs...)
    idxs = clusterDistances(D; kwargs...)
    if dim==2
        X[:, idxs] # Orders columns by default
    elseif dim==1
        X[idxs, :]
    end
end

export clusterDistances
export clusterPairwise
export clusterReorder