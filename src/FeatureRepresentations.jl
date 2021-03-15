using Catch22


function featureRepresentation(X, featureFunc::Function)
    F = featureFunc(X)
end
export featureRepresentation