# using BoxCoxTrans
import YeoJohnsonTrans as YJT
using Tullio




function YJT.transform(X::AbstractArray, λs::Vector)
    Y = deepcopy(X)
    for j in size(X, 1)
        Y[j, :] = YJT.transform(X[j, :], λs[j])
    end
    return Y
end

"""
Yeo-Johnson transformation according to a matrix X
"""
function yeojohnson(X::AbstractArray)
    λs = Array{Float64, 1}(undef, size(X, 1))
    for (i, x) ∈ eachrow(X) |> enumerate
        λs[i], details = YJT.lambda(x; interval=(-10.0, 10.0))
    end
    return Y -> YJT.transform(Y, λs)
end
export yeojohnson
