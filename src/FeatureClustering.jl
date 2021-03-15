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

function clusterColumns(X::AbstractMatrix, distanceMetric=Distances.CorrDist(); kwargs...)
    D = pairwise(distanceMetric, X, dims=2)
    clusterDistances(D; kwargs...)
end

function clusterReorder(X::AbstractMatrix, distanceMetric=Distances.CorrDist(); kwargs...)
    idxs = clusterColumns(X, distanceMetric; kwargs...)
    X[:, idxs] # Orders columns. Transpose beforehand you lazy sod
end

export clusterReorder