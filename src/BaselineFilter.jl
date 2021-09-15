
"""
A crude method for finding integer-valued rows of an array
"""
function findintegerfeatures(X::AbstractArray, N=10, prec=3)
    mask = fill(false, size(X, 1))
    for (i, f) ∈ enumerate(eachrow(X))
        σ = std(f)
        fr = round.(f./σ, sigdigits=prec)
        mask[i] = length(unique(fr)) < N
    end
    return mask
end
export findintegerfeatures

_nointegerfeatures(X::AbstractArray, args...) = Y->Y[.!findintegerfeatures(X, args...), :]
nointegerfeatures(X, args...) = _nointegerfeatures(X, args...)(X)
export _nointegerfeatures, nointegerfeatures


"""
Can we remove any perfectly correlated features?
"""
function findredundantfeatures(X::AbstractArray)
    Σ = cor(X, dims=2)
    for r ∈ 1:size(Σ, 1)
        cfs = findall(r == 1.0) # Features perfectly correlated to this one
        cfs = cfs[cfs .!= r] # Don't count the one we're looking at
        for cf ∈ cfs
            Σ[cf, :] .= NaN
            Σ[:, cf] .= NaN
        end
    end
    idxs = isnan.(diag(Σ))
end
export findredundantfeatures

_noredundantfeatures(X::AbstractArray, args...) = Y->Y[.!findredundantfeatures(X, args...), :]
noredundantfeatures(X, args...) = _noredundantfeatures(X, args...)(X)
export _noredundantfeatures, noredundantfeatures
