using Catch22
using NamedArrays


function features(X, featureFunc)
    F = featureFunc(X)
end