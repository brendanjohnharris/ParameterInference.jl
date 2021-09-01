""""""
function heuristic(Fₕ, Fₗ=zeros(size(Fₕ)); transform=baselinetransform(Fₕ, Fₗ), filter=baselinefilter(Fₕ, Fₗ))
    Fₗ = Fₗ |> transform |> filter
    Fₕ = Fₕ |> transform |> filter
    scale = intervalscaled(Fₕ, Fₗ, filter=identity, transform=identity)
    Fₗ = Fₗ |> scale
    Fₕ = Fₕ |> scale
    function correction(F)
        Σₕ² = StatsBase.cov(Array(Fₕ), dims=2)
        𝐧 = sum(abs.(Array(Σₕ²)), dims=2)
        N⁻¹ = FeatureMatrix(inv(sqrt(Diagonal(𝐧[:]))), getnames(Fₕ))
        return FeatureMatrix(N⁻¹*F, getnames(F))
    end
    return F -> F |> transform |> filter |> scale |> correction
end
export heuristic
