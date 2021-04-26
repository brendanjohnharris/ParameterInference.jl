# ------------------------------------------------------------------------------------------------ #
#          Run a statistical test to compare the distributions of features in two matrices         #
# ------------------------------------------------------------------------------------------------ #
@userplot StatFeatureFilter
@recipe function f(P::StatFeatureFilter; tail=:firstlarger, test=TailedTestOfVariance)
    F₁ = P.args[1]
    F₂ = P.args[2]

    p = testFeatureDistributions(Fₗ, Fₕ, test, tail=tail)

    featureviolin(F₁, F₂)
end

