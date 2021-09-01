""""""
function heuristic(Fₕ, Fₗ=zeros(size(Fₕ)); kwargs...)
    scale = intervalscaled(Fₕ, Fₗ, kwargs...)
    Fₗ = Fₗ |> scale
    Fₕ = Fₕ |> scale
    function correction(F)
        Σₕ² = StatsBase.cov(Array(Fₕ), dims=2)
        𝐧 = sum(abs.(Array(Σₕ²)), dims=2)
        𝐧[𝐧 .== 0] .= Inf
        N⁻¹ = FeatureMatrix(inv(sqrt(Diagonal(𝐧[:]))), getnames(Fₕ))
        return FeatureMatrix(N⁻¹*F, getnames(F))
    end
    return F -> F |> scale |> correction
end
export heuristic
