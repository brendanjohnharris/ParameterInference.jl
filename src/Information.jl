using TransferEntropy
using Tullio
using Entropies


MI(x::AbstractVector, y) = TransferEntropy.mutualinfo(vec(x), vec(y), RectangularBinning(10); base=ℯ)
MI(X::AbstractArray, Y) = @tullio I[i, j] := MI(X[:, i], Y[:, j]) # Columns to match cor and corspearman
MI(X) = MI(X, X)
export MI

H(x) = Entropies.genentropy(probabilities(vec(x), 10), q=1.0, base=ℯ)

MI_norm(x::AbstractVector, y)  = MI(x, y)/(H(x) + H(y) - MI(x, y))
MI_norm(X::AbstractArray, Y) = @tullio I[i, j] := MI_norm(X[:, i], Y[:, j])
MI_norm(X) = MI_norm(X, X)
export MI_norm

# Something doesn't seem quite right...
