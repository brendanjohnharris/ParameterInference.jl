""""""
function heuristic(Fₕ, Fₗ=zeros(size(Fₕ)); transform=baselinetransform, filter=baselinefilter)
    transform = transform(Fₕ, Fₗ)
    Fₗ = Fₗ |> transform
    Fₕ = Fₕ |> transform
    filter = filter(Fₕ, Fₗ)
    Fₗ = Fₗ |> filter
    Fₕ = Fₕ |> filter
    scale = intervalscaled(Fₕ, Fₗ, filter=identity, transform=identity)
    Fₗ = Fₗ |> scale
    Fₕ = Fₕ |> scale
    function correction(F)
        Σₕ² = StatsBase.cov(Array(Fₕ), dims=2)
        𝐧 = sum(abs.(Array(Σₕ²)), dims=2)
        𝐧[𝐧 .== 0] .= Inf
        N⁻¹ = FeatureMatrix(inv(sqrt(Diagonal(𝐧[:]))), getnames(Fₕ))
        return FeatureMatrix(N⁻¹*F, getnames(F))
    end
    return F -> F |> transform |> filter |> scale |> correction
end
export heuristic
