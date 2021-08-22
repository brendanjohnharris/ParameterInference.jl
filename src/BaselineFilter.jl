
"""
A crude method for finding integer-valued rows of an array
"""
function findintegerfeatures(X::AbstractArray, N=10, prec=3)
    mask = fill(false, size(X, 1))
    for (i, f) ∈ enumerate(eachrow(Y))
        σ = std(f)
        fr = round.(f./σ, sigdigits=prec)
        mask[i] = length(unique(fr)) < N
    end
    return mask
end
export findintegerfeatures

_nointegerfeatures(X::AbstractArray, args...) = X->X[.!findintegerfeatures(X, args...), :]
nointegerfeatures(X, args...) = _nointegerfeatures(X, args...)(X)
export _nointegerfeatures, nointegerfeatures
