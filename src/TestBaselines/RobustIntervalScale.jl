""""""
function robustintervalscale(Fₕ, Fₗ=zeros(size(Fₕ)))
    σₗ = mapslices(iqr, Fₗ, dims=2) |> Array
    σₕ = mapslices(iqr, Fₕ, dims=2) |> Array
    function scale(F)
        σ = mapslices(iqr, F, dims=2) |> Array
        σᵢ = [rampOn(0, 1, σₗ[i], σₕ[i])(σ[i]) for i ∈ 1:length(σ)] # No need to worry about means?
        idxs = vec(isnan.(σᵢ) .| isnan.(σ))
        σᵢ[idxs] .= 0
        σ[idxs] .= Inf
        return F.*σᵢ./σ
    end
    return scale
end
export robustintervalscale
