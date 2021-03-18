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

function clusterColumns(X::AbstractMatrix, distanceMetric=Distances.CorrDist(); dim=2, kwargs...)
    D = pairwise(distanceMetric, X, dims=dim)
    clusterDistances(D; kwargs...)
end

function clusterReorder(X::AbstractMatrix, distanceMetric=Distances.CorrDist(); kwargs...)
    idxs = clusterColumns(X, distanceMetric; kwargs...)
    X[:, idxs] # Orders columns by default
end

function clusterReorder(X::AbstractMatrix, D::AbstractMatrix; kwargs...)
    idxs = clusterDistances(D; kwargs...)
    X[:, idxs] # Orders columns by default
end

export clusterDistances
export clusterColumns
export clusterReorder